"""Screening pipeline: parse -> normalise -> SBERT -> score -> rank -> persist.

Runs in a background thread so the UI can show the processing screen. Progress is
published through a small in-memory registry keyed by job id.
"""
from __future__ import annotations

import hashlib
import threading
import traceback

from app.database import SessionLocal
from app.models import Candidate, Job, SkillEvidence
from app.services import embeddings, ml_models, scoring
from app.services import extractor as X
from app.services.parser import parse_resume

STAGES = ["Extracting resume information", "Understanding skills",
          "Analyzing experience", "Semantic matching", "Ranking candidates"]

_progress: dict[int, dict] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ progress
def get_progress(job_id: int) -> dict:
    with _lock:
        return dict(_progress.get(job_id, {"state": "idle", "percent": 0, "stage": "",
                                           "done": 0, "total": 0, "stages_done": []}))


def _set(job_id: int, **kw):
    with _lock:
        cur = _progress.setdefault(job_id, {"state": "running", "percent": 0, "stage": "",
                                            "done": 0, "total": 0, "stages_done": [],
                                            "errors": []})
        cur.update(kw)


# ------------------------------------------------------------------ helpers
def job_text(job) -> str:
    """The text SBERT compares resumes against: title + description + requirements."""
    return " ".join([job.title or "", job.description or "",
                     "Requirements: " + ", ".join(r.skill for r in job.requirements)])


def job_key(job) -> str:
    return hashlib.sha1(job_text(job).encode("utf-8")).hexdigest()


def _evidence_rows(cand_id: int, job, skills_detail: dict) -> list[SkillEvidence]:
    """One evidence row per requirement, plus SBERT related skills for the gaps."""
    missing = [r.skill for r in job.requirements if r.skill not in skills_detail]
    related = embeddings.related_skills(missing, sorted(skills_detail))
    rows = []
    for r in job.requirements:
        info = skills_detail.get(r.skill)
        if info:
            rows.append(SkillEvidence(
                candidate_id=cand_id, skill=r.skill, requirement_type=r.category,
                status="Strong" if info["count"] >= 2 else "Partial",
                mentions=info["count"], matched_surface="; ".join(info["surfaces"]),
                evidence=info["evidence"]))
        else:
            near, sim = related.get(r.skill, ("", 0.0))
            rows.append(SkillEvidence(
                candidate_id=cand_id, skill=r.skill, requirement_type=r.category,
                status="Missing", related_skill=near, related_score=sim))
    return rows


# ------------------------------------------------------------------ screening
def start_screening(job_id: int, file_paths: list[tuple[str, str]]):
    """Kick off screening in a background thread. file_paths: [(stored_path, original_name)]"""
    _set(job_id, state="running", percent=2, stage=STAGES[0], done=0,
         total=len(file_paths), stages_done=[], errors=[])
    t = threading.Thread(target=_run, args=(job_id, file_paths), daemon=True)
    t.start()
    return t


def _run(job_id: int, file_paths: list[tuple[str, str]]):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            _set(job_id, state="error", errors=["Job not found."])
            return

        # 1. parse every resume
        parsed, paths, errors = [], [], []
        for i, (path, original) in enumerate(file_paths, 1):
            try:
                parsed.append(parse_resume(path, original))
                paths.append(path)
            except Exception as exc:                          # noqa: BLE001
                errors.append(f"{original}: {exc}")
            _set(job_id, done=i, percent=2 + int(40 * i / max(1, len(file_paths))),
                 stage=STAGES[0], errors=errors)

        if not parsed:
            _set(job_id, state="error", percent=100,
                 errors=errors or ["None of the files contained readable text."])
            return

        # 2-3. skills and experience were extracted during parsing
        _set(job_id, stages_done=STAGES[:3], stage=STAGES[3], percent=45)

        # 4. SBERT semantic matching (batched across every resume)
        jt, key = job_text(job), job_key(job)
        sem_raw = embeddings.resume_similarity([p["resume_text"] for p in parsed], jt)
        sem = embeddings.calibrate(sem_raw)
        ctx = [(p["projects"] + " " + p["work_experience"]).strip() or p["resume_text"]
               for p in parsed]
        proj = embeddings.calibrate(embeddings.resume_similarity(ctx, jt))
        _set(job_id, stages_done=STAGES[:4], stage=STAGES[4], percent=78)

        # 5. score, explain, persist
        rel_model = ml_models.get_relevance_model()
        cat_model = ml_models.get_category_model()
        for i, p in enumerate(parsed):
            skills = set(p["skills_detail"])
            sc = scoring.composite(job, skills, p["years_experience"],
                                   p["education_level"], float(sem[i]), float(proj[i]))
            cand = Candidate(
                job_id=job.id, file_name=p["file_name"], stored_path=paths[i],
                name=p["name"], email=p["email"], phone=p["phone"],
                current_title=p["current_title"], summary=p["summary"],
                skills=p["skills"], education=p["education"], degree=p["degree"],
                education_level=p["education_level"], field_of_study=p["field_of_study"],
                years_experience=p["years_experience"], experience_source=p["experience_source"],
                certifications=p["certifications"], projects=p["projects"],
                work_experience=p["work_experience"], resume_text=p["resume_text"][:20000],
                skill_score=sc.skills, experience_score=sc.experience,
                education_score=sc.education, semantic_score=sc.semantic,
                semantic_raw=float(sem_raw[i]), semantic_key=key,
                project_score=sc.project, overall_score=sc.overall,
                passes_filters=sc.passes, filter_reason=sc.reason,
                stage="AI Screened" if sc.passes else "Applied",
                predicted_category=cat_model.predict(p["resume_text"]) if cat_model else "")
            if rel_model:
                cand.ml_label, cand.ml_score = rel_model.predict(dict(
                    skill=sc.skills, experience=sc.experience, education=sc.education,
                    project=sc.project, semantic=sc.semantic,
                    domain=1.0 if cand.predicted_category == job.category else 0.0))
            db.add(cand)
            db.flush()
            db.add_all(_evidence_rows(cand.id, job, p["skills_detail"]))
            _set(job_id, percent=78 + int(20 * (i + 1) / len(parsed)))

        db.commit()
        rerank(db, job.id)
        _set(job_id, state="done", percent=100, stage="Completed",
             stages_done=STAGES, errors=errors)
    except embeddings.SBERTNotAvailable as exc:
        _set(job_id, state="error", errors=[str(exc)])
    except Exception:                                          # noqa: BLE001
        _set(job_id, state="error", errors=[traceback.format_exc()[-600:]])
    finally:
        db.close()


# ------------------------------------------------------------------ ranking
def rerank(db, job_id: int):
    """Sort by (passes hard filters, overall score). Filtered-out candidates rank
    below eligible ones but stay visible - nothing is deleted."""
    cands = db.query(Candidate).filter(Candidate.job_id == job_id).all()
    cands.sort(key=lambda c: (c.passes_filters, c.overall_score), reverse=True)
    for i, c in enumerate(cands, 1):
        c.rank = i
    db.commit()
    return cands


def rescore_job(db, job_id: int):
    """Recompute every score after the recruiter changes weights, filters or
    requirements.

    - Weights or filters changed: pure arithmetic on stored values, instant.
    - Requirements changed: the job text changed, so SBERT similarity is
      recomputed; the evidence rows are rebuilt from the stored resume text,
      keeping the quoted sentences and Strong / Partial / Missing statuses.
    """
    job = db.get(Job, job_id)
    cands = db.query(Candidate).filter(Candidate.job_id == job_id).all()
    if not cands:
        return []

    key = job_key(job)
    stale = [c for c in cands if c.semantic_key != key]
    if stale:
        raw = embeddings.resume_similarity([c.resume_text for c in stale], job_text(job))
        for c, r in zip(stale, raw):
            c.semantic_raw = float(r)
            c.semantic_score = float(embeddings.calibrate(r))
            c.semantic_key = key

    for c in cands:
        detail = X.normalized_skills(c.resume_text or "")
        sc = scoring.composite(job, set(detail), c.years_experience, c.education_level,
                               c.semantic_score, c.project_score)
        (c.skill_score, c.experience_score, c.education_score, c.overall_score,
         c.passes_filters, c.filter_reason) = (sc.skills, sc.experience, sc.education,
                                               sc.overall, sc.passes, sc.reason)
        db.query(SkillEvidence).filter(SkillEvidence.candidate_id == c.id).delete()
        db.add_all(_evidence_rows(c.id, job, detail))
    db.commit()
    return rerank(db, job_id)

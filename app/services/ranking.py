"""Screening pipeline: parse -> normalise -> SBERT -> score -> rank -> persist.

Runs in a background thread so the UI can show the processing screen. Progress is
published through a small in-memory registry keyed by job id.
"""
from __future__ import annotations

import hashlib
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import PARSE_WORKERS
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


def summarise_errors(failures: list[tuple[str, str]]) -> list[str]:
    """One line per distinct problem instead of one per file, e.g.
    "72 files (a.pdf, b.pdf, c.pdf and 69 more): scanned PDF ... OCR is not available ..."."""
    groups: dict[str, list[str]] = {}
    for name, msg in failures:
        groups.setdefault(msg, []).append(name)
    out = []
    for msg, names in groups.items():
        if len(names) == 1:
            out.append(f"{names[0]}: {msg}")
        else:
            shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
            out.append(f"{len(names)} files ({shown}): {msg}")
    return out


def match_requirements(skills_detail: dict, resume_text: str, job) -> dict:
    """Make every job requirement findable in skills_detail under its own name.

    Requirements typed by hand may not be spelled like the ontology ("java",
    "J2EE", "springboot") or may not be in it at all ("Hybris"). The first are
    mapped to their canonical skill, the second are matched as a phrase in the
    resume text. Without this, such a requirement is "missing" for everyone.
    """
    detail = dict(skills_detail)
    for r in job.requirements:
        if r.skill in detail:
            continue
        canon = X.canonical_skill(r.skill)
        if canon in detail:
            detail[r.skill] = detail[canon]
            continue
        hit = X.custom_skill_hits(resume_text, r.skill)
        if hit:
            detail[r.skill] = hit
    return detail


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

        # 1. parse every resume. Scanned resumes need OCR (a few seconds a page),
        #    so files are parsed in parallel; results keep the upload order.
        results: dict[int, dict] = {}
        failures: list[tuple[str, str]] = []
        errors: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as pool:
            futures = {pool.submit(parse_resume, path, original): (i, original)
                       for i, (path, original) in enumerate(file_paths)}
            for fut in as_completed(futures):
                i, original = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:                      # noqa: BLE001
                    failures.append((original, str(exc)))
                    errors = summarise_errors(failures)
                done += 1
                _set(job_id, done=done, percent=2 + int(40 * done / max(1, len(file_paths))),
                     stage=STAGES[0], errors=list(errors))
        parsed = [results[i] for i in sorted(results)]
        paths = [file_paths[i][0] for i in sorted(results)]

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
            p["skills_detail"] = match_requirements(p["skills_detail"], p["resume_text"], job)
            p["skills"] = "; ".join(sorted(p["skills_detail"]))
            skills = set(p["skills_detail"])
            sc = scoring.composite(job, skills, p["years_experience"],
                                   p["education_level"], float(sem[i]), float(proj[i]),
                                   doc_type=p.get("doc_type", "resume"))
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
    # Equal scores (common when a job has only one or two requirements) are
    # ordered by semantic similarity to the job, then skill coverage, then experience.
    cands.sort(key=lambda c: (c.passes_filters, round(c.overall_score, 4), c.semantic_raw or 0.0,
                              c.skill_score or 0.0, c.years_experience or 0.0), reverse=True)
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
        detail = match_requirements(X.normalized_skills(c.resume_text or ""), c.resume_text or "", job)
        c.skills = "; ".join(sorted(detail))
        sc = scoring.composite(job, set(detail), c.years_experience, c.education_level,
                               c.semantic_score, c.project_score,
                               doc_type=X.document_type(c.resume_text or ""))
        (c.skill_score, c.experience_score, c.education_score, c.overall_score,
         c.passes_filters, c.filter_reason) = (sc.skills, sc.experience, sc.education,
                                               sc.overall, sc.passes, sc.reason)
        db.query(SkillEvidence).filter(SkillEvidence.candidate_id == c.id).delete()
        db.add_all(_evidence_rows(c.id, job, detail))
    db.commit()
    return rerank(db, job_id)

"""Screening pipeline: parse -> normalise -> SBERT -> score -> rank -> persist.

Two ways an application arrives:
  - a candidate applies on HireX               -> add_application()   (one at a time)
  - a recruiter bulk-uploads outside resumes   -> start_screening()   (background thread)
Both score through services/analysis.py, so the numbers are identical.
"""
from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import PARSE_WORKERS
from app.database import SessionLocal
from app.models import Candidate, Job, SkillEvidence
from app.services import analysis, embeddings
from app.services import extractor as X
from app.services.analysis import job_key, job_text, match_requirements  # noqa: F401 (re-exported)
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


def summarise_errors(failures: list[tuple[str, str]]) -> list[str]:
    """One line per distinct problem instead of one per file."""
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


# ------------------------------------------------------------------ building rows
def evidence_rows(cand_id: int, job, skills_detail: dict) -> list[SkillEvidence]:
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


def apply_scores(cand: Candidate, a: dict):
    """Copy an analysis result onto a Candidate row (new or re-scored)."""
    sc = a["score"]
    cand.skills = "; ".join(sorted(a["skills_detail"]))
    cand.skill_score, cand.experience_score = sc.skills, sc.experience
    cand.education_score, cand.semantic_score = sc.education, sc.semantic
    cand.semantic_raw, cand.semantic_key = a["sem_raw"], a["key"]
    cand.project_score, cand.internship_score = sc.project, sc.internship
    cand.practical_score, cand.overall_score = sc.practical, sc.overall
    cand.passes_filters, cand.filter_reason = sc.passes, sc.reason
    cand.fit_label, cand.fit_reason = a["fit_label"], a["fit_reason"][:255]
    cand.practical_detail_json = json.dumps(a["practical"]["detail"])
    cand.ml_label, cand.ml_score = a["ml_label"], a["ml_score"]
    if a["predicted_category"]:
        cand.predicted_category = a["predicted_category"]


def build_candidate(job, p: dict, a: dict, stored_path: str, **extra) -> Candidate:
    found = p.get("practical") or {}
    cand = Candidate(
        job_id=job.id, file_name=p["file_name"], stored_path=stored_path,
        name=p["name"], email=p["email"], phone=p["phone"],
        current_title=p["current_title"], summary=p["summary"],
        education=p["education"], degree=p["degree"],
        education_level=p["education_level"], field_of_study=p["field_of_study"],
        years_experience=p["years_experience"], experience_source=p["experience_source"],
        certifications=p["certifications"], projects=p["projects"],
        work_experience=p["work_experience"], resume_text=p["resume_text"][:20000],
        internship_months=0 if any(e.get("kind") == "job" for e in found.get("internships", []))
        else found.get("internship_months", 0),
        internship_count=sum(1 for e in found.get("internships", []) if e.get("kind") != "job"),
        project_count=len(found.get("projects", [])),
        practical_json=json.dumps(found),
        **extra)
    apply_scores(cand, a)
    if not cand.stage:
        cand.stage = "AI Screened" if a["score"].passes else "Applied"
    return cand


def add_application(db, job, p: dict, stored_path: str, **extra) -> Candidate:
    """Score one resume, store it as an application and re-rank the job."""
    a = analysis.analyse(job, p)
    extra.setdefault("stage", "AI Screened" if a["score"].passes else "Applied")
    cand = build_candidate(job, p, a, stored_path, **extra)
    db.add(cand)
    db.flush()
    db.add_all(evidence_rows(cand.id, job, a["skills_detail"]))
    db.commit()
    rerank(db, job.id)
    return cand


# ------------------------------------------------------------------ bulk screening
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

        # 2-3. skills, experience, internships and projects were extracted during parsing
        _set(job_id, stages_done=STAGES[:3], stage=STAGES[3], percent=45)

        # 4. SBERT semantic matching (batched across every resume)
        sem_raw = embeddings.resume_similarity([p["resume_text"] for p in parsed], job_text(job))
        _set(job_id, stages_done=STAGES[:4], stage=STAGES[4], percent=70)

        # 5. score, explain, persist
        for i, p in enumerate(parsed):
            a = analysis.analyse(job, p, sem_raw=float(sem_raw[i]))
            cand = build_candidate(job, p, a, paths[i], applied_via="bulk",
                                   stage="AI Screened" if a["score"].passes else "Applied")
            db.add(cand)
            db.flush()
            db.add_all(evidence_rows(cand.id, job, a["skills_detail"]))
            _set(job_id, percent=70 + int(28 * (i + 1) / len(parsed)))

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
    """Sort by (eligible, overall score). Applicants who fail a hard filter or are
    'Not eligible' rank below everyone eligible but stay visible - nothing is deleted."""
    cands = db.query(Candidate).filter(Candidate.job_id == job_id).all()
    # Equal scores are ordered by semantic similarity, then skill coverage, then experience.
    cands.sort(key=lambda c: (c.eligible, round(c.overall_score, 4), c.semantic_raw or 0.0,
                              c.skill_score or 0.0, c.years_experience or 0.0), reverse=True)
    for i, c in enumerate(cands, 1):
        c.rank = i
    db.commit()
    return cands


def rescore_job(db, job_id: int):
    """Recompute every score after the recruiter changes weights, filters or
    requirements, from the stored resume text (no file is read again)."""
    job = db.get(Job, job_id)
    cands = db.query(Candidate).filter(Candidate.job_id == job_id).all()
    if not cands:
        return []

    key = job_key(job)
    stale = [c for c in cands if c.semantic_key != key]
    raw = {}
    if stale:
        sims = embeddings.resume_similarity([c.resume_text for c in stale], job_text(job))
        raw = {c.id: float(r) for c, r in zip(stale, sims)}

    for c in cands:
        text = c.resume_text or ""
        p = {"resume_text": text, "skills_detail": X.normalized_skills(text),
             "years_experience": c.years_experience, "education_level": c.education_level,
             "practical": c.practical, "doc_type": X.document_type(text)}
        a = analysis.analyse(job, p, sem_raw=raw.get(c.id, c.semantic_raw))
        apply_scores(c, a)
        db.query(SkillEvidence).filter(SkillEvidence.candidate_id == c.id).delete()
        db.add_all(evidence_rows(c.id, job, a["skills_detail"]))
    db.commit()
    return rerank(db, job_id)

"""Score one parsed resume against one job. No database code in here.

Used by every path that scores a resume, so they can never disagree:
  - the recruiter's bulk upload           (services/ranking.py)
  - the candidate's "Check my match"      (routers/portal.py)
  - the candidate's application           (routers/portal.py)
  - re-scoring after the recruiter edits requirements or weights
"""
from __future__ import annotations

import hashlib

from app.services import embeddings, ml_models, practical, scoring
from app.services import extractor as X


# ------------------------------------------------------------------ job text
def job_text(job) -> str:
    """The text SBERT compares resumes against: title + description + requirements."""
    return " ".join([job.title or "", job.description or "",
                     "Requirements: " + ", ".join(r.skill for r in job.requirements)])


def job_key(job) -> str:
    return hashlib.sha1(job_text(job).encode("utf-8")).hexdigest()


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


# ------------------------------------------------------------------ practical work
def practical_for(job, found: dict, use_sbert: bool = True) -> dict:
    """Internships + projects against this job (see services/practical.py)."""
    sim_fn = None
    if use_sbert:
        jt = job_text(job)

        def sim_fn(texts):
            return [float(v) for v in embeddings.calibrate(embeddings.resume_similarity(texts, jt))]
    return practical.score(job.requirements, found or {}, sim_fn)


# ------------------------------------------------------------------ one resume
def analyse(job, p: dict, sem_raw: float | None = None, use_sbert: bool = True) -> dict:
    """Everything needed to store and explain one application.

    p: parser output (services/parser.parse_resume), including p["practical"].
    sem_raw: raw SBERT cosine if it was already computed in a batch.
    """
    text = p.get("resume_text", "")
    if sem_raw is None:
        sem_raw = float(embeddings.resume_similarity([text], job_text(job))[0]) if use_sbert else 0.0
    sem = float(embeddings.calibrate(sem_raw))

    detail = match_requirements(p.get("skills_detail") or {}, text, job)
    pr = practical_for(job, p.get("practical") or {}, use_sbert)
    sc = scoring.composite(job, set(detail), p.get("years_experience", 0.0),
                           p.get("education_level", 0), sem, pr,
                           doc_type=p.get("doc_type", "resume"))

    cat_model = ml_models.get_category_model()
    predicted = cat_model.predict(text) if cat_model else ""
    label, why = scoring.fit_label(sc, predicted, job.category)

    ml_label, ml_score = 0, 0.0
    rel_model = ml_models.get_relevance_model()
    if rel_model:
        ml_label, ml_score = rel_model.predict(dict(
            skill=sc.skills, experience=sc.experience, education=sc.education,
            project=sc.project, semantic=sc.semantic,
            domain=1.0 if predicted == job.category else 0.0))

    return {"skills_detail": detail, "score": sc, "sem_raw": float(sem_raw),
            "practical": pr, "predicted_category": predicted,
            "fit_label": label, "fit_reason": why,
            "ml_label": ml_label, "ml_score": ml_score, "key": job_key(job)}


def preview(job, p: dict, a: dict) -> dict:
    """JSON-safe summary for the 'Check my match' screen (candidate side).
    Uses the same explanation as the recruiter sees, worded for the candidate."""
    sc = a["score"]
    detail = a["skills_detail"]
    req_rows = []
    for r in job.requirements:
        info = detail.get(r.skill)
        req_rows.append({"skill": r.skill, "category": r.category, "found": bool(info),
                         "evidence": (info or {}).get("evidence", "")})
    pct = sc.as_percent()
    parts = [{"label": "Skills", "match": pct["skills"], "weight": round(job.weights["skills"] * 100)},
             {"label": "Experience", "match": pct["experience"], "weight": round(job.weights["experience"] * 100)},
             {"label": "Practical work", "match": pct["practical"], "weight": round(job.weights["practical"] * 100)},
             {"label": "Education", "match": pct["education"], "weight": round(job.weights["education"] * 100)},
             {"label": "Overall fit", "match": pct["semantic"], "weight": round(job.weights["semantic"] * 100)}]
    return {"overall": pct["overall"], "label": a["fit_label"], "reason": a["fit_reason"],
            "parts": [x for x in parts if x["weight"] > 0],
            "requirements": req_rows,
            "practical_notes": a["practical"]["detail"]["notes"],
            "penalty": sc.detail.get("penalty", ""),
            "filter_reason": "" if sc.passes else sc.reason,
            "name": p.get("name", ""), "years": p.get("years_experience", 0.0),
            "degree": p.get("degree", ""),
            "internships": sum(1 for e in (p.get("practical") or {}).get("internships", []) if e.get("kind") != "job"),
            "projects": len((p.get("practical") or {}).get("projects", [])),
            "used_ocr": bool(p.get("used_ocr")), "hidden_text": bool(p.get("hidden_text"))}

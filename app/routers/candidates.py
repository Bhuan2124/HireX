"""Ranked results, the 'Why selected?' page, comparison, stage tracking and search."""
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import STAGES
from app.database import get_db
from app.models import Candidate, Job, SkillEvidence
from app.security import require_login
from app.services import scoring
from app.templating import templates

router = APIRouter(tags=["candidates"])


def _job(db, job_id, rid):
    job = db.get(Job, job_id)
    return job if job and job.recruiter_id == rid else None


@router.get("/jobs/{job_id}/results", response_class=HTMLResponse)
def results(job_id: int, request: Request, view: str = "all", top: int = 50,
            db: Session = Depends(get_db)):
    rid = require_login(request)
    job = _job(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)

    cands = (db.query(Candidate).filter(Candidate.job_id == job_id)
             .order_by(Candidate.rank.asc()).all())

    if view == "strong":
        shown = [c for c in cands if c.overall_score >= 0.70 and c.passes_filters]
    elif view == "gap":
        shown = [c for c in cands if c.skill_score < 0.60]
    elif view == "shortlisted":
        shown = [c for c in cands if c.stage == "Shortlisted"]
    elif view == "filtered":
        shown = [c for c in cands if not c.passes_filters]
    else:
        shown = [c for c in cands if c.passes_filters]
    shown = shown[:top]

    missing_counts = {}
    for c in cands:
        for e in c.evidence:
            if e.status == "Missing":
                missing_counts[e.skill] = missing_counts.get(e.skill, 0) + 1
    common_gaps = sorted(missing_counts.items(), key=lambda kv: -kv[1])[:6]

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "job": job, "candidates": shown, "all_count": len(cands),
         "view": view, "top": top, "stages": STAGES,
         "eligible": sum(1 for c in cands if c.passes_filters),
         "filtered_out": sum(1 for c in cands if not c.passes_filters),
         "common_gaps": common_gaps})


@router.get("/candidates/{cand_id}", response_class=HTMLResponse)
def candidate_detail(cand_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    cand = db.get(Candidate, cand_id)
    if not cand:
        return RedirectResponse("/dashboard", status_code=303)
    job = _job(db, cand.job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)

    score = scoring.Score(skills=cand.skill_score, experience=cand.experience_score,
                          education=cand.education_score, semantic=cand.semantic_score,
                          project=cand.project_score, overall=cand.overall_score,
                          passes=cand.passes_filters, reason=cand.filter_reason)
    ev = sorted(cand.evidence, key=lambda e: ({"Strong": 0, "Partial": 1, "Missing": 2}[e.status],
                                              e.skill))
    payload = scoring.explain(cand, job, score, ev)

    skills = set(cand.skill_list)
    uplift = sorted(
        ({"skill": e.skill,
          "gain": scoring.uplift_if_added(job, skills, e.skill, score)}
         for e in ev if e.status == "Missing"),
        key=lambda d: -d["gain"])[:3]

    # rank of the candidate immediately above, for the "why not higher" line
    above = (db.query(Candidate).filter(Candidate.job_id == job.id,
                                        Candidate.rank == max(1, cand.rank - 1)).first())

    return templates.TemplateResponse(
        "candidate.html",
        {"request": request, "job": job, "c": cand, "evidence": ev,
         "explain": payload, "uplift": uplift, "above": above, "stages": STAGES})


@router.post("/candidates/{cand_id}/stage")
def set_stage(cand_id: int, request: Request, stage: str = Form(...),
              back: str = Form("results"), db: Session = Depends(get_db)):
    rid = require_login(request)
    cand = db.get(Candidate, cand_id)
    if cand and _job(db, cand.job_id, rid) and stage in STAGES:
        cand.stage = stage
        db.commit()
    if back == "detail":
        return RedirectResponse(f"/candidates/{cand_id}", status_code=303)
    return RedirectResponse(f"/jobs/{cand.job_id}/results", status_code=303)


@router.get("/jobs/{job_id}/compare", response_class=HTMLResponse)
def compare(job_id: int, request: Request, ids: str = "", db: Session = Depends(get_db)):
    rid = require_login(request)
    job = _job(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)
    id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()][:4]
    chosen = (db.query(Candidate).filter(Candidate.id.in_(id_list),
                                         Candidate.job_id == job_id).all()
              if id_list else [])
    chosen.sort(key=lambda c: c.rank)
    req_skills = [r.skill for r in job.requirements]
    matrix = {}
    for c in chosen:
        have = {e.skill: e.status for e in c.evidence}
        matrix[c.id] = {s: have.get(s, "Missing") for s in req_skills}
    return templates.TemplateResponse(
        "compare.html", {"request": request, "job": job, "candidates": chosen,
                         "req_skills": req_skills, "matrix": matrix})


@router.get("/jobs/{job_id}/search", response_class=HTMLResponse)
def search(job_id: int, request: Request, q: str = "", db: Session = Depends(get_db)):
    """Search the screened pool by skill, title or name."""
    rid = require_login(request)
    job = _job(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)
    cands = db.query(Candidate).filter(Candidate.job_id == job_id).order_by(
        Candidate.rank.asc()).all()
    term = q.strip().lower()
    hits = [c for c in cands
            if term and (term in (c.skills or "").lower()
                         or term in (c.current_title or "").lower()
                         or term in (c.name or "").lower())] if term else []
    return templates.TemplateResponse(
        "search.html", {"request": request, "job": job, "q": q, "hits": hits,
                        "total": len(cands)})


@router.get("/candidates/{cand_id}/file")
def download_resume(cand_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    cand = db.get(Candidate, cand_id)
    if not cand or not _job(db, cand.job_id, rid):
        return RedirectResponse("/dashboard", status_code=303)
    return FileResponse(cand.stored_path, filename=cand.file_name)

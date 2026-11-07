"""Ranked results, the 'Why selected?' page, comparison, stage tracking and search."""
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import REJECT_REASONS, STAGES
from app.database import get_db
from app.models import Candidate, Job, SkillEvidence
from app.security import require_login
from app.services import platform, scoring
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
    platform.release_due_jobs(db)

    cands = (db.query(Candidate).filter(Candidate.job_id == job_id)
             .order_by(Candidate.rank.asc()).all())

    if view == "strong":
        shown = [c for c in cands if c.overall_score >= 0.70 and c.eligible]
    elif view == "gap":
        shown = [c for c in cands if c.skill_score < 0.60]
    elif view == "shortlisted":
        shown = [c for c in cands if c.stage in ("Shortlisted", "Interview", "Selected")]
    elif view == "flagged":
        shown = [c for c in cands if c.duplicate_of_id]
    elif view == "filtered":
        shown = [c for c in cands if not c.eligible]
    else:
        shown = [c for c in cands if c.eligible]
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
         "eligible": sum(1 for c in cands if c.eligible),
         "filtered_out": sum(1 for c in cands if not c.eligible),
         "flagged": sum(1 for c in cands if c.duplicate_of_id),
         "zone": min(job.zone_size, sum(1 for c in cands if c.eligible)),
         "reject_reasons": REJECT_REASONS,
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
                          project=cand.project_score or 0.0, internship=cand.internship_score or 0.0,
                          practical=cand.practical_score or 0.0, overall=cand.overall_score,
                          passes=cand.passes_filters, reason=cand.filter_reason)
    ev = sorted(cand.evidence, key=lambda e: ({"Strong": 0, "Partial": 1, "Missing": 2}[e.status],
                                              e.skill))
    payload = scoring.explain(cand, job, score, ev)

    uplift = sorted(
        ({"skill": e.skill,
          "gain": round((platform.overall_if_added(job, cand, e.skill) - cand.overall_score) * 100, 1)}
         for e in ev if e.status == "Missing"),
        key=lambda d: -d["gain"])[:3]
    dup = db.get(Candidate, cand.duplicate_of_id) if cand.duplicate_of_id else None

    # rank of the candidate immediately above, for the "why not higher" line
    above = (db.query(Candidate).filter(Candidate.job_id == job.id,
                                        Candidate.rank == max(1, cand.rank - 1)).first())

    return templates.TemplateResponse(
        "candidate.html",
        {"request": request, "job": job, "c": cand, "evidence": ev,
         "explain": payload, "uplift": uplift, "above": above, "stages": STAGES,
         "reject_reasons": REJECT_REASONS, "dup": dup, "practical": cand.practical,
         "pdetail": cand.practical_detail})


@router.post("/candidates/{cand_id}/stage")
def set_stage(cand_id: int, request: Request, stage: str = Form(...),
              back: str = Form("results"), reason: str = Form(""), note: str = Form(""),
              interview_at: str = Form(""), db: Session = Depends(get_db)):
    rid = require_login(request)
    cand = db.get(Candidate, cand_id)
    job = _job(db, cand.job_id, rid) if cand else None
    if not cand or not job:
        return RedirectResponse("/dashboard", status_code=303)
    if stage in STAGES:
        platform.record_decision(db, cand, job, stage,
                                 reason if reason in REJECT_REASONS else "", note, interview_at)
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


# ------------------------------------------------------------------ bulk decisions
@router.post("/jobs/{job_id}/shortlist-zone")
def shortlist_zone(job_id: int, request: Request, db: Session = Depends(get_db)):
    """Shortlist the top N eligible applicants (the interview zone) in one click."""
    rid = require_login(request)
    job = _job(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)
    cands = (db.query(Candidate).filter(Candidate.job_id == job_id)
             .order_by(Candidate.rank.asc()).all())
    for c in [x for x in cands if x.eligible][:job.zone_size]:
        if c.stage in ("Applied", "AI Screened"):
            platform.record_decision(db, c, job, "Shortlisted")
    db.commit()
    return RedirectResponse(f"/jobs/{job_id}/results?view=shortlisted", status_code=303)


@router.post("/jobs/{job_id}/reject-rest")
def reject_rest(job_id: int, request: Request, reason: str = Form("Not a fit for this role"),
                note: str = Form(""), db: Session = Depends(get_db)):
    """Reject everyone who has no decision yet, with one reason. Each candidate
    still gets their own screening feedback ('Why not me?')."""
    rid = require_login(request)
    job = _job(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)
    reason = reason if reason in REJECT_REASONS else "Other"
    for c in db.query(Candidate).filter(Candidate.job_id == job_id,
                                        Candidate.stage.in_(["Applied", "AI Screened"])).all():
        platform.record_decision(db, c, job, "Rejected", reason, note)
    db.commit()
    return RedirectResponse(f"/jobs/{job_id}/results", status_code=303)

"""Dashboard, job creation, AI requirement extraction and recruiter preferences."""
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import CATEGORIES, EDUCATION_LEVELS, REQ_CATEGORIES, REQ_WEIGHT
from app.database import get_db
from app.models import Candidate, Job, Requirement
from app.security import require_login
from app.services import jd_extractor, ranking
from app.templating import templates

router = APIRouter(tags=["jobs"])


def _job_or_403(db, job_id, rid):
    job = db.get(Job, job_id)
    if job is None or job.recruiter_id != rid:
        return None
    return job


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    jobs = db.query(Job).filter(Job.recruiter_id == rid).order_by(Job.created_at.desc()).all()
    cards = []
    for j in jobs:
        cands = db.query(Candidate).filter(Candidate.job_id == j.id).all()
        shortlisted = sum(1 for c in cands if c.stage == "Shortlisted")
        best = max((c.overall_score for c in cands), default=0.0)
        cards.append({"job": j, "total": len(cands), "shortlisted": shortlisted,
                      "best": round(best * 100)})
    return templates.TemplateResponse("dashboard.html",
                                      {"request": request, "cards": cards})


# ------------------------------------------------------------------ create job
@router.get("/jobs/new", response_class=HTMLResponse)
def new_job_form(request: Request):
    require_login(request)
    return templates.TemplateResponse("job_new.html",
                                      {"request": request, "categories": CATEGORIES})


@router.post("/jobs/new")
def create_job(request: Request, title: str = Form(...), company: str = Form(""),
               location: str = Form(""), employment_type: str = Form("Full-time"),
               category: str = Form("INFORMATION-TECHNOLOGY"),
               description: str = Form(""), db: Session = Depends(get_db)):
    rid = require_login(request)
    job = Job(recruiter_id=rid, title=title.strip(), company=company.strip(),
              location=location.strip(), employment_type=employment_type,
              category=category, description=description.strip())
    db.add(job)
    db.commit()

    # --- AI requirement extraction from the job description ---
    found = jd_extractor.summarise(job.description, job.title)
    for r in found["requirements"]:
        db.add(Requirement(job_id=job.id, skill=r["skill"], category=r["category"],
                           importance=r["importance"], source="extracted"))
    job.min_experience = found["min_experience"]
    job.min_education = found["min_education"]
    db.commit()
    return RedirectResponse(f"/jobs/{job.id}/requirements", status_code=303)


# ------------------------------------------------------- review requirements
@router.get("/jobs/{job_id}/requirements", response_class=HTMLResponse)
def requirements_page(job_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    job = _job_or_403(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        "job_requirements.html",
        {"request": request, "job": job, "req_categories": REQ_CATEGORIES,
         "education_levels": list(EDUCATION_LEVELS.keys()),
         "evidence": {r.skill: r.source for r in job.requirements}})


@router.post("/jobs/{job_id}/requirements/add")
def add_requirement(job_id: int, request: Request, skill: str = Form(...),
                    category: str = Form("Must Have"), db: Session = Depends(get_db)):
    rid = require_login(request)
    job = _job_or_403(db, job_id, rid)
    if job and skill.strip():
        exists = any(r.skill.lower() == skill.strip().lower() for r in job.requirements)
        if not exists:
            db.add(Requirement(job_id=job.id, skill=skill.strip(), category=category,
                               importance=REQ_WEIGHT.get(category, 1.0), source="manual"))
            db.commit()
    return RedirectResponse(f"/jobs/{job_id}/requirements", status_code=303)


@router.post("/jobs/{job_id}/requirements/{req_id}/delete")
def delete_requirement(job_id: int, req_id: int, request: Request,
                       db: Session = Depends(get_db)):
    rid = require_login(request)
    job = _job_or_403(db, job_id, rid)
    req = db.get(Requirement, req_id)
    if job and req and req.job_id == job.id:
        db.delete(req)
        db.commit()
    return RedirectResponse(f"/jobs/{job_id}/requirements", status_code=303)


@router.post("/jobs/{job_id}/requirements/update")
async def update_requirements(job_id: int, request: Request, db: Session = Depends(get_db)):
    """Save edited categories/importances, hard filters and weights in one go."""
    rid = require_login(request)
    job = _job_or_403(db, job_id, rid)
    if not job:
        return RedirectResponse("/dashboard", status_code=303)
    form = await request.form()

    for r in job.requirements:
        cat = form.get(f"cat_{r.id}")
        if cat in REQ_CATEGORIES:
            r.category = cat
        imp = form.get(f"imp_{r.id}")
        if imp:
            try:
                r.importance = max(0.1, min(2.0, float(imp)))
            except ValueError:
                pass

    try:
        job.min_experience = float(form.get("min_experience") or 0)
    except ValueError:
        job.min_experience = 0.0
    job.min_education = form.get("min_education") or "Any"

    weights = {k: float(form.get(f"w_{k}") or 0) for k in
               ("skills", "experience", "education", "semantic")}
    total = sum(weights.values()) or 1.0
    job.w_skills = weights["skills"] / total
    job.w_experience = weights["experience"] / total
    job.w_education = weights["education"] / total
    job.w_semantic = weights["semantic"] / total
    db.commit()

    # if candidates already exist, re-score them with the new settings
    if db.query(Candidate).filter(Candidate.job_id == job.id).count():
        ranking.rescore_job(db, job.id)
        return RedirectResponse(f"/jobs/{job.id}/results", status_code=303)
    return RedirectResponse(f"/jobs/{job.id}/upload", status_code=303)


@router.post("/jobs/{job_id}/delete")
def delete_job(job_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    job = _job_or_403(db, job_id, rid)
    if job:
        db.delete(job)
        db.commit()
    return RedirectResponse("/dashboard", status_code=303)

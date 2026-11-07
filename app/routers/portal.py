"""Candidate side of HireX.

/find-jobs                         search open jobs (role, city, field, type, level)
/find-jobs/{id}                    job details + "Check my match"
/find-jobs/{id}/check   (POST)     read the resume and score it - NOT an application yet
/find-jobs/{id}/check/{check_id}   the match result: %, label, found / missing skills
/find-jobs/{id}/apply/{check_id}   apply with that same resume (once, no resubmission)
/my/applications                   every job applied to, with its status
/my/applications/{id}              status, decision, and after the deadline: rank + "Why not me?"
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import (ALLOWED_EXTENSIONS, CATEGORIES, EXPERIENCE_LEVELS, MAX_UPLOAD_MB,
                        REQ_CATEGORIES, UPLOAD_DIR)
from app.database import get_db
from app.models import Candidate, CandidateUser, Job, MatchCheck, Requirement
from app.security import require_candidate
from app.services import analysis, platform, ranking
from app.services.parser import parse_resume
from app.templating import templates

router = APIRouter(tags=["candidate portal"])

JOB_TYPES = ["Full-time", "Part-time", "Internship", "Contract"]


def _my_application(db, cid: int, job_id: int):
    return db.query(Candidate).filter(Candidate.candidate_user_id == cid,
                                      Candidate.job_id == job_id).first()


# ------------------------------------------------------------------ search
@router.get("/find-jobs", response_class=HTMLResponse)
def find_jobs(request: Request, q: str = "", city: str = "", category: str = "",
              type: str = "", level: str = "", closed: int = 0, db: Session = Depends(get_db)):
    cid = require_candidate(request)
    platform.release_due_jobs(db)
    query = db.query(Job)
    term = q.strip()
    if term:
        like = f"%{term}%"
        skill_jobs = db.query(Requirement.job_id).filter(Requirement.skill.ilike(like))
        query = query.filter(or_(Job.title.ilike(like), Job.company.ilike(like),
                                 Job.description.ilike(like), Job.id.in_(skill_jobs)))
    if city.strip():
        query = query.filter(Job.location.ilike(f"%{city.strip()}%"))
    if category in CATEGORIES:
        query = query.filter(Job.category == category)
    if type in JOB_TYPES:
        query = query.filter(Job.employment_type == type)
    if level in EXPERIENCE_LEVELS:
        query = query.filter(Job.experience_level.in_([level, "Any"]))
    jobs = query.order_by(Job.created_at.desc()).all()
    if not closed:
        jobs = [j for j in jobs if j.is_open]
    jobs.sort(key=lambda j: (not j.is_open, j.apply_until or j.created_at))
    applied = {c.job_id: c.id for c in db.query(Candidate).filter(Candidate.candidate_user_id == cid)}
    return templates.TemplateResponse(
        "find_jobs.html",
        {"request": request, "nav": "find", "jobs": jobs, "applied": applied,
         "q": q, "city": city, "category": category, "type": type, "level": level,
         "closed": closed, "categories": CATEGORIES, "types": JOB_TYPES, "levels": EXPERIENCE_LEVELS})


# ------------------------------------------------------------------ one job
def _job_page(request, db, job, cid, error="", status_code=200):
    app_ = _my_application(db, cid, job.id)
    groups = [(cat, [r for r in job.requirements if r.category == cat]) for cat in REQ_CATEGORIES]
    return templates.TemplateResponse(
        "job_view.html",
        {"request": request, "nav": "find", "job": job, "application": app_,
         "groups": [(c, rows) for c, rows in groups if rows], "error": error,
         "applicants": db.query(Candidate).filter(Candidate.job_id == job.id).count(),
         "max_mb": MAX_UPLOAD_MB}, status_code=status_code)


@router.get("/find-jobs/{job_id}", response_class=HTMLResponse)
def job_view(job_id: int, request: Request, db: Session = Depends(get_db)):
    cid = require_candidate(request)
    job = db.get(Job, job_id)
    if not job:
        return RedirectResponse("/find-jobs", status_code=303)
    return _job_page(request, db, job, cid)


@router.post("/find-jobs/{job_id}/check")
def check_match(job_id: int, request: Request, resume: UploadFile = File(...),
                db: Session = Depends(get_db)):
    """Read the resume and score it against this job. Nothing is sent to the
    recruiter until the candidate presses Apply."""
    cid = require_candidate(request)
    job = db.get(Job, job_id)
    if not job:
        return RedirectResponse("/find-jobs", status_code=303)
    if not job.is_open:
        return _job_page(request, db, job, cid, "Applications for this job have closed.", 400)
    if _my_application(db, cid, job.id):
        return RedirectResponse(f"/find-jobs/{job_id}", status_code=303)

    ext = Path(resume.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _job_page(request, db, job, cid, "Please upload a PDF, Word (.docx), text or image file.", 400)
    folder = Path(UPLOAD_DIR) / "candidates" / str(cid)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4().hex[:12]}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(resume.file, out)
    if dest.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        dest.unlink(missing_ok=True)
        return _job_page(request, db, job, cid, f"That file is over {MAX_UPLOAD_MB} MB.", 400)

    try:
        p = parse_resume(str(dest), resume.filename or dest.name)
    except Exception as exc:                                   # noqa: BLE001
        dest.unlink(missing_ok=True)
        return _job_page(request, db, job, cid,
                         f"We couldn't read that resume ({exc}). Try a text-based PDF or a .docx file.", 400)
    a = analysis.analyse(job, p)
    check = MatchCheck(candidate_user_id=cid, job_id=job.id, file_name=p["file_name"],
                       stored_path=str(dest), parsed_json=json.dumps(p),
                       analysis_json=json.dumps(analysis.preview(job, p, a)))
    db.add(check)
    db.commit()
    return RedirectResponse(f"/find-jobs/{job_id}/check/{check.id}", status_code=303)


def _check(db, cid, job_id, check_id):
    c = db.get(MatchCheck, check_id)
    if not c or c.candidate_user_id != cid or c.job_id != job_id:
        return None
    return c


@router.get("/find-jobs/{job_id}/check/{check_id}", response_class=HTMLResponse)
def check_result(job_id: int, check_id: int, request: Request, db: Session = Depends(get_db)):
    cid = require_candidate(request)
    job = db.get(Job, job_id)
    check = _check(db, cid, job_id, check_id) if job else None
    if not check:
        return RedirectResponse(f"/find-jobs/{job_id}", status_code=303)
    if check.used or _my_application(db, cid, job_id):
        return RedirectResponse(f"/find-jobs/{job_id}", status_code=303)
    return templates.TemplateResponse(
        "match_result.html",
        {"request": request, "nav": "find", "job": job, "check": check,
         "m": json.loads(check.analysis_json or "{}")})


@router.post("/find-jobs/{job_id}/apply/{check_id}")
def apply(job_id: int, check_id: int, request: Request, confirm: str = Form(""),
          db: Session = Depends(get_db)):
    cid = require_candidate(request)
    job = db.get(Job, job_id)
    check = _check(db, cid, job_id, check_id) if job else None
    if not check:
        return RedirectResponse(f"/find-jobs/{job_id}", status_code=303)
    existing = _my_application(db, cid, job_id)
    if existing:                                   # one application per job, no resubmission
        return RedirectResponse(f"/my/applications/{existing.id}", status_code=303)
    if not job.is_open:
        return _job_page(request, db, job, cid, "Applications for this job closed before you applied.", 400)
    if confirm != "yes":
        return RedirectResponse(f"/find-jobs/{job_id}/check/{check_id}", status_code=303)

    user = db.get(CandidateUser, cid)
    p = json.loads(check.parsed_json)
    cand = ranking.add_application(db, job, p, check.stored_path,
                                   candidate_user_id=cid, applied_via="platform")
    # the applicant's own account details are more reliable than what OCR read
    cand.name = user.name or cand.name
    cand.email = cand.email or user.email
    cand.phone = cand.phone or user.phone
    check.used = True
    db.commit()
    platform.flag_duplicate(db, cand)
    platform.notify(db, "recruiter", job.recruiter_id,
                    f"New application from {cand.name} for {job.title} ({cand.match_percent}% match).",
                    f"/candidates/{cand.id}")
    platform.notify(db, "candidate", cid,
                    f"Your application for {job.title} was submitted. Results come out after the deadline.",
                    f"/my/applications/{cand.id}")
    db.commit()
    return RedirectResponse(f"/my/applications/{cand.id}?new=1", status_code=303)


# ------------------------------------------------------------------ my applications
@router.get("/my/applications", response_class=HTMLResponse)
def my_applications(request: Request, db: Session = Depends(get_db)):
    cid = require_candidate(request)
    platform.release_due_jobs(db)
    apps = (db.query(Candidate).filter(Candidate.candidate_user_id == cid)
            .order_by(Candidate.created_at.desc()).all())
    rows = [{"c": c, "job": c.job, "status": platform.candidate_status(c, c.job)} for c in apps]
    return templates.TemplateResponse("my_applications.html",
                                      {"request": request, "nav": "mine", "rows": rows})


@router.get("/my/applications/{cand_id}", response_class=HTMLResponse)
def my_application(cand_id: int, request: Request, new: int = 0, db: Session = Depends(get_db)):
    cid = require_candidate(request)
    platform.release_due_jobs(db)
    c = db.get(Candidate, cand_id)
    if not c or c.candidate_user_id != cid:
        return RedirectResponse("/my/applications", status_code=303)
    job = c.job
    released = job.results_released
    ev = sorted(c.evidence, key=lambda e: ({"Strong": 0, "Partial": 1, "Missing": 2}[e.status], e.skill))
    return templates.TemplateResponse(
        "application.html",
        {"request": request, "nav": "mine", "c": c, "job": job, "new": new,
         "status": platform.candidate_status(c, job), "released": released,
         "evidence": ev, "parts": platform.components(job, c),
         "fb": platform.feedback(db, c, job) if released else None,
         "pdetail": c.practical_detail})

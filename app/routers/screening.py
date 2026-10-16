"""Bulk resume upload, the AI processing screen and its progress API."""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, UPLOAD_DIR
from app.database import get_db
from app.models import Candidate, Job
from app.security import require_login
from app.services import embeddings, ml_models, ranking
from app.templating import templates

router = APIRouter(tags=["screening"])


@router.get("/jobs/{job_id}/upload", response_class=HTMLResponse)
def upload_page(job_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    job = db.get(Job, job_id)
    if not job or job.recruiter_id != rid:
        return RedirectResponse("/dashboard", status_code=303)
    existing = db.query(Candidate).filter(Candidate.job_id == job_id).count()
    return templates.TemplateResponse(
        "upload.html", {"request": request, "job": job, "existing": existing,
                        "sbert": embeddings.status(), "ml": ml_models.status()})


@router.post("/jobs/{job_id}/upload")
async def upload_resumes(job_id: int, request: Request,
                         files: list[UploadFile] = File(...),
                         db: Session = Depends(get_db)):
    rid = require_login(request)
    job = db.get(Job, job_id)
    if not job or job.recruiter_id != rid:
        return RedirectResponse("/dashboard", status_code=303)

    job_dir = Path(UPLOAD_DIR) / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    saved, skipped = [], []
    for uf in files:
        ext = Path(uf.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(f"{uf.filename} (unsupported type)")
            continue
        dest = job_dir / f"{uuid.uuid4().hex[:12]}{ext}"
        with dest.open("wb") as out:
            shutil.copyfileobj(uf.file, out)
        if dest.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            dest.unlink(missing_ok=True)
            skipped.append(f"{uf.filename} (over {MAX_UPLOAD_MB} MB)")
            continue
        saved.append((str(dest), uf.filename or dest.name))

    if not saved:
        return templates.TemplateResponse(
            "upload.html", {"request": request, "job": job, "existing": 0,
                            "error": "No usable files. " + "; ".join(skipped),
                            "sbert": embeddings.status(), "ml": ml_models.status()},
            status_code=400)

    # replace any previous run for this job
    db.query(Candidate).filter(Candidate.job_id == job_id).delete()
    db.commit()

    ranking.start_screening(job_id, saved)
    return RedirectResponse(f"/jobs/{job_id}/processing", status_code=303)


@router.get("/jobs/{job_id}/processing", response_class=HTMLResponse)
def processing_page(job_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    job = db.get(Job, job_id)
    if not job or job.recruiter_id != rid:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        "processing.html", {"request": request, "job": job, "stages": ranking.STAGES})


@router.get("/api/jobs/{job_id}/progress")
def progress(job_id: int, request: Request):
    require_login(request)
    return JSONResponse(ranking.get_progress(job_id))

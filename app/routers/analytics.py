"""Recruiter analytics: funnel, score distribution, skill supply and gaps."""
from collections import Counter

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import STAGES
from app.database import get_db
from app.models import Candidate, Job
from app.security import require_login
from app.services import embeddings, ml_models
from app.templating import templates

router = APIRouter(tags=["analytics"])


@router.get("/jobs/{job_id}/analytics", response_class=HTMLResponse)
def job_analytics(job_id: int, request: Request, db: Session = Depends(get_db)):
    rid = require_login(request)
    job = db.get(Job, job_id)
    if not job or job.recruiter_id != rid:
        return RedirectResponse("/dashboard", status_code=303)

    cands = db.query(Candidate).filter(Candidate.job_id == job_id).all()
    funnel = [(s, sum(1 for c in cands if c.stage == s)) for s in STAGES]

    buckets = [("90-100%", 0.90), ("80-89%", 0.80), ("70-79%", 0.70),
               ("60-69%", 0.60), ("50-59%", 0.50), ("below 50%", 0.0)]
    dist = []
    for i, (label, low) in enumerate(buckets):
        high = 1.01 if i == 0 else buckets[i - 1][1]
        dist.append((label, sum(1 for c in cands if low <= c.overall_score < high)))
    peak = max((n for _, n in dist), default=1) or 1

    supply = Counter()
    gaps = Counter()
    for c in cands:
        for s in c.skill_list:
            supply[s] += 1
        for e in c.evidence:
            if e.status == "Missing":
                gaps[e.skill] += 1

    avg = round(sum(c.overall_score for c in cands) / len(cands) * 100) if cands else 0
    return templates.TemplateResponse(
        "analytics.html",
        {"request": request, "job": job, "total": len(cands), "funnel": funnel,
         "dist": dist, "peak": peak, "avg": avg,
         "top_skills": supply.most_common(10), "top_gaps": gaps.most_common(10),
         "eligible": sum(1 for c in cands if c.passes_filters),
         "sbert": embeddings.status(), "ml": ml_models.status()})

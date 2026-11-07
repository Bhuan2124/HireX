"""The bell icon: notifications for whichever account is logged in."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Notification
from app.security import current_candidate_id, current_recruiter_id
from app.services import platform
from app.templating import templates

router = APIRouter(tags=["notifications"])


def _who(request: Request):
    rid = current_recruiter_id(request)
    if rid:
        return "recruiter", rid
    cid = current_candidate_id(request)
    if cid:
        return "candidate", cid
    return None, None


@router.get("/notifications", response_class=HTMLResponse)
def notifications(request: Request, db: Session = Depends(get_db)):
    kind, uid = _who(request)
    if not kind:
        return RedirectResponse("/login", status_code=303)
    platform.release_due_jobs(db)
    items = (db.query(Notification).filter(Notification.user_type == kind, Notification.user_id == uid)
             .order_by(Notification.created_at.desc(), Notification.id.desc()).limit(100).all())
    unread_ids = {n.id for n in items if not n.is_read}
    for n in items:
        n.is_read = True
    db.commit()
    return templates.TemplateResponse("notifications.html",
                                      {"request": request, "items": items, "unread_ids": unread_ids})

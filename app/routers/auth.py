"""Registration, login and logout for both kinds of account.

/register?role=recruiter   recruiter account (posts jobs)
/register?role=candidate   candidate account (applies for jobs)
"""
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CandidateUser, Recruiter
from app.security import hash_password, log_in, verify_password
from app.templating import templates

router = APIRouter(tags=["auth"])

ROLES = ("recruiter", "candidate")
HOME = {"recruiter": "/dashboard", "candidate": "/find-jobs"}


def _role(value: str) -> str:
    return value if value in ROLES else "candidate"


def _model(role: str):
    return Recruiter if role == "recruiter" else CandidateUser


@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request, role: str = "candidate"):
    return templates.TemplateResponse("register.html", {"request": request, "role": _role(role)})


@router.post("/register")
def register(request: Request, role: str = Form("candidate"), name: str = Form(...),
             email: str = Form(...), company: str = Form(""), phone: str = Form(""),
             city: str = Form(""), password: str = Form(...),
             db: Session = Depends(get_db)):
    role = _role(role)
    Model = _model(role)
    email = email.strip().lower()
    ctx = {"request": request, "role": role, "name": name, "email": email,
           "company": company, "phone": phone, "city": city}
    if db.query(Model).filter(Model.email == email).first():
        return templates.TemplateResponse(
            "register.html", {**ctx, "error": "That email is already registered."}, status_code=400)
    if len(password) < 6:
        return templates.TemplateResponse(
            "register.html", {**ctx, "error": "Password must be at least 6 characters."}, status_code=400)
    if role == "recruiter":
        user = Recruiter(name=name.strip(), email=email, company=company.strip(),
                         password_hash=hash_password(password))
    else:
        user = CandidateUser(name=name.strip(), email=email, phone=phone.strip(),
                             city=city.strip(), password_hash=hash_password(password))
    db.add(user)
    db.commit()
    log_in(request, role, user)
    return RedirectResponse(HOME[role], status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, role: str = "candidate"):
    return templates.TemplateResponse("login.html", {"request": request, "role": _role(role)})


@router.post("/login")
def login(request: Request, role: str = Form("candidate"), email: str = Form(...),
          password: str = Form(...), db: Session = Depends(get_db)):
    role = _role(role)
    Model = _model(role)
    user = db.query(Model).filter(Model.email == email.strip().lower()).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html", {"request": request, "role": role, "email": email,
                           "error": "Incorrect email or password."}, status_code=401)
    log_in(request, role, user)
    return RedirectResponse(HOME[role], status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)

"""Recruiter registration, login and logout."""
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Recruiter
from app.security import hash_password, verify_password
from app.templating import templates

router = APIRouter(tags=["auth"])


@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@router.post("/register")
def register(request: Request, name: str = Form(...), email: str = Form(...),
             company: str = Form(""), password: str = Form(...),
             db: Session = Depends(get_db)):
    email = email.strip().lower()
    if db.query(Recruiter).filter(Recruiter.email == email).first():
        return templates.TemplateResponse(
            "register.html", {"request": request, "error": "That email is already registered.",
                              "name": name, "email": email, "company": company}, status_code=400)
    if len(password) < 6:
        return templates.TemplateResponse(
            "register.html", {"request": request, "error": "Password must be at least 6 characters.",
                              "name": name, "email": email, "company": company}, status_code=400)
    user = Recruiter(name=name.strip(), email=email, company=company.strip(),
                     password_hash=hash_password(password))
    db.add(user)
    db.commit()
    request.session["recruiter_id"] = user.id
    request.session["recruiter_name"] = user.name
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...),
          db: Session = Depends(get_db)):
    user = db.query(Recruiter).filter(Recruiter.email == email.strip().lower()).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Incorrect email or password.",
                           "email": email}, status_code=401)
    request.session["recruiter_id"] = user.id
    request.session["recruiter_name"] = user.name
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)

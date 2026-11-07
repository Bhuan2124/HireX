"""Password hashing (PBKDF2-HMAC-SHA256) and session helpers.

No external crypto dependency: the standard library is enough for this project.

The session holds ONE logged-in account at a time:
  recruiter_id  -> a recruiter is logged in
  candidate_id  -> a candidate (job seeker) is logged in
Logging in as one clears the other.
"""
import base64
import hashlib
import hmac
import os

from fastapi import HTTPException, Request, status

ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_b64, dk_b64 = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except Exception:
        return False


def _redirect(to: str):
    return HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": to})


# ------------------------------------------------------------------ recruiter
def current_recruiter_id(request: Request):
    return request.session.get("recruiter_id")


def require_login(request: Request) -> int:
    """Recruiter pages. (Name kept from the recruiter-only version.)"""
    rid = current_recruiter_id(request)
    if not rid:
        raise _redirect("/login?role=candidate" if request.session.get("candidate_id")
                        else "/login?role=recruiter")
    return rid


require_recruiter = require_login


# ------------------------------------------------------------------ candidate
def current_candidate_id(request: Request):
    return request.session.get("candidate_id")


def require_candidate(request: Request) -> int:
    cid = current_candidate_id(request)
    if not cid:
        raise _redirect("/login?role=candidate")
    return cid


def log_in(request: Request, role: str, user):
    request.session.clear()
    if role == "recruiter":
        request.session["recruiter_id"] = user.id
        request.session["recruiter_name"] = user.name
    else:
        request.session["candidate_id"] = user.id
        request.session["candidate_name"] = user.name

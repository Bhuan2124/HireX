"""Password hashing (PBKDF2-HMAC-SHA256) and session helpers.

No external crypto dependency: the standard library is enough for this project.
"""
import hashlib, hmac, os, base64
from fastapi import Request, HTTPException, status

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


def current_recruiter_id(request: Request):
    return request.session.get("recruiter_id")


def require_login(request: Request) -> int:
    rid = current_recruiter_id(request)
    if not rid:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER,
                            headers={"Location": "/login"})
    return rid

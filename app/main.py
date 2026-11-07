"""HireX - FastAPI application entry point.

Start the server from the project root:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import os
import sys

from app.config import BASE_DIR, SECRET_KEY
from app.database import init_db
from app.routers import analytics, auth, candidates, jobs, notifications, portal, screening
from app.services import embeddings, ml_models, ocr
from app.templating import templates

app = FastAPI(title="HireX", description="Explainable AI hiring platform for recruiters and candidates",
              version="4.0")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=60 * 60 * 24 * 7)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(screening.router)
app.include_router(candidates.router)
app.include_router(analytics.router)
app.include_router(portal.router)
app.include_router(notifications.router)


@app.on_event("startup")
def on_startup():
    init_db()
    try:
        embeddings.init()               # load Sentence-BERT once, before any request
    except embeddings.SBERTNotAvailable as exc:
        print("\n" + "=" * 70)
        print("HireX cannot start: Sentence-BERT is not available.")
        print(str(exc))
        print("=" * 70 + "\n")
        sys.exit(1)
    ml_models.status()                  # load trained models if present
    if os.getenv("HIREX_SEED_DEMO") == "1":
        try:                            # demo accounts on a fresh deployment
            from app.services.demo import seed_if_empty
            seed_if_empty()
        except Exception as exc:        # noqa: BLE001 - never block startup over demo data
            print("Demo data not loaded:", exc)
    print("HireX ready" + ("" if os.getenv("PORT") else " at http://127.0.0.1:8000"))


@app.get("/api/status")
def api_status():
    """Quick health check: which model is running, on which device."""
    return {"sbert": embeddings.status(), "ml": ml_models.status(), "ocr": ocr.status()}


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    if request.session.get("recruiter_id"):
        return RedirectResponse("/dashboard", status_code=303)
    if request.session.get("candidate_id"):
        return RedirectResponse("/find-jobs", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})


@app.exception_handler(404)
def not_found(request: Request, exc):
    return templates.TemplateResponse("error.html",
                                      {"request": request, "code": 404,
                                       "message": "Page not found"}, status_code=404)

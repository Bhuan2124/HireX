"""Demo data for reviews (used by tools/seed_demo.py and by HIREX_SEED_DEMO=1)

Fill HireX with demo data so both sides can be shown in a review.

    python tools/seed_demo.py            # adds the demo accounts, jobs and applications
    python tools/seed_demo.py --reset    # deletes hirex.db first (start clean)
    HIREX_SEED_DEMO=1                    # deployed: loads it on the first start of an empty database

Everyone's password is  demo123

Recruiter   hr@acme.com
Candidates  ananya@demo.com   2 internships + 3 projects        (fresher, strong)
            sneha@demo.com    projects only, no internship      (fresher)
            rahul@demo.com    1 short internship + 2 projects   (fresher)
            vikram@demo.com   5 years as a Java developer       (experienced)
            priya@demo.com    fitness coach                     (wrong field -> "Not eligible")
            anu@demo.com      near-copy of Ananya's resume      (flagged as a possible copy)

Jobs        "Java Developer (Freshers)"   all six have applied, deadline already PASSED
                                          -> open My applications as any candidate to see
                                             the final rank and "Why not me?"
            "Backend Engineer"            open for 7 days, nobody has applied yet
                                          -> log in as a candidate and try "Check my match"
            "Fitness Trainer"             open, so Priya has a job that fits her

The resumes are in tools/demo_resumes/. Sentence-BERT must be installed (same as
for running the app), because applications are scored exactly like real ones.
"""
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from app.config import BASE_DIR, UPLOAD_DIR, WEIGHTS_BY_LEVEL
from app.database import SessionLocal, init_db
from app.models import CandidateUser, Job, Recruiter, Requirement
from app.security import hash_password
from app.services import embeddings, jd_extractor, platform, ranking
from app.services.parser import parse_resume

PASSWORD = "demo123"
RESUMES = BASE_DIR / "tools" / "demo_resumes"

CANDIDATES = [  # email, name, city, resume file
    ("ananya@demo.com", "Ananya Rao", "Bengaluru", "ananya_rao.txt"),
    ("sneha@demo.com", "Sneha Kulkarni", "Pune", "sneha_kulkarni.txt"),
    ("rahul@demo.com", "Rahul Menon", "Kochi", "rahul_menon.txt"),
    ("vikram@demo.com", "Vikram Iyer", "Bengaluru", "vikram_iyer.txt"),
    ("priya@demo.com", "Priya Nair", "Mysuru", "priya_nair.txt"),
    ("anu@demo.com", "Anu Reddy", "Hyderabad", "anu_reddy.txt"),
]

JOBS = [
    dict(title="Java Developer (Freshers)", company="Acme Technologies", location="Bengaluru",
         employment_type="Full-time", category="INFORMATION-TECHNOLOGY", experience_level="Fresher",
         openings=1, days=7,
         description="We are hiring a fresher Java developer to build REST APIs.\n"
                     "Required: Java, Spring Boot.\nPreferred: SQL.\nDocker is a plus.\n"
                     "Education: Bachelor's degree in Computer Science or related field."),
    dict(title="Backend Engineer", company="Acme Technologies", location="Bengaluru (Hybrid)",
         employment_type="Full-time", category="INFORMATION-TECHNOLOGY", experience_level="Any",
         openings=2, days=7,
         description="Build backend services in Java or Python.\nRequired: Java, REST API, SQL.\n"
                     "Preferred: Spring Boot, Docker, Kubernetes.\nAWS experience is a plus."),
    dict(title="Fitness Trainer", company="Acme Wellness", location="Mysuru",
         employment_type="Full-time", category="FITNESS", experience_level="Experienced",
         openings=1, days=10,
         description="Certified personal trainer for our gym.\nRequired: personal training, strength training.\n"
                     "Preferred: nutrition, yoga. Group classes experience is a plus."),
]


def get_or_create(db, Model, email, **fields):
    obj = db.query(Model).filter(Model.email == email).first()
    if obj:
        return obj, False
    obj = Model(email=email, password_hash=hash_password(PASSWORD), **fields)
    db.add(obj)
    db.commit()
    return obj, True


def make_job(db, recruiter, spec):
    job = db.query(Job).filter(Job.recruiter_id == recruiter.id, Job.title == spec["title"]).first()
    if job:
        return job, False
    w = WEIGHTS_BY_LEVEL[spec["experience_level"]]
    job = Job(recruiter_id=recruiter.id, title=spec["title"], company=spec["company"],
              location=spec["location"], employment_type=spec["employment_type"],
              category=spec["category"], description=spec["description"],
              experience_level=spec["experience_level"], openings=spec["openings"],
              apply_until=(datetime.now() + timedelta(days=spec["days"])).replace(second=0, microsecond=0),
              w_skills=w["skills"], w_experience=w["experience"], w_practical=w["practical"],
              w_education=w["education"], w_semantic=w["semantic"])
    db.add(job)
    db.commit()
    found = jd_extractor.summarise(job.description, job.title)
    for r in found["requirements"]:
        db.add(Requirement(job_id=job.id, skill=r["skill"], category=r["category"],
                           importance=r["importance"], source="extracted"))
    job.min_experience = 0.0 if job.experience_level == "Fresher" else found["min_experience"]
    job.min_education = found["min_education"]
    db.commit()
    db.refresh(job)
    return job, True


def apply(db, job, user, resume_file):
    from app.models import Candidate
    if db.query(Candidate).filter(Candidate.job_id == job.id,
                                  Candidate.candidate_user_id == user.id).first():
        return None
    folder = Path(UPLOAD_DIR) / "candidates" / str(user.id)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / resume_file
    shutil.copy(RESUMES / resume_file, dest)
    p = parse_resume(str(dest), resume_file)
    cand = ranking.add_application(db, job, p, str(dest), candidate_user_id=user.id, applied_via="platform")
    cand.name, cand.email = user.name, cand.email or user.email
    db.commit()
    platform.flag_duplicate(db, cand)
    platform.notify(db, "recruiter", job.recruiter_id,
                    f"New application from {cand.name} for {job.title} ({cand.match_percent}% match).",
                    f"/candidates/{cand.id}")
    db.commit()
    return cand


def seed(verbose: bool = True):
    """Create the demo accounts, jobs and applications (skips what already exists)."""
    init_db()
    embeddings.init()
    say = print if verbose else (lambda *a, **k: None)
    db = SessionLocal()
    try:
        rec, _ = get_or_create(db, Recruiter, "hr@acme.com", name="Meera Shah", company="Acme Technologies")
        users = {}
        for email, name, city, resume in CANDIDATES:
            users[email], _ = get_or_create(db, CandidateUser, email, name=name, city=city)

        jobs = {}
        for spec in JOBS:
            jobs[spec["title"]], created = make_job(db, rec, spec)
            say(("Created" if created else "Exists ") + f" job: {spec['title']}")

        java = jobs["Java Developer (Freshers)"]
        for email, name, city, resume in CANDIDATES:
            c = apply(db, java, users[email], resume)
            if c:
                say(f"  {name:15} applied: {c.match_percent:3}%  {c.fit_label}"
                      + ("  [possible copy]" if c.duplicate_of_id else ""))

        # Close the Java job now so its results can be shown in the demo, and make
        # a couple of decisions so candidates see different outcomes.
        if not java.results_released:
            ranked = ranking.rerank(db, java.id)
            eligible = [c for c in ranked if c.eligible]
            if eligible:
                platform.record_decision(db, eligible[0], java, "Interview",
                                         interview_at=(datetime.now() + timedelta(days=5)).strftime("%d %b, 11:00 AM, Google Meet"),
                                         note="Your internship work on REST APIs stood out.")
            for c in ranked:
                if c.stage in ("Applied", "AI Screened") and c not in eligible[:java.zone_size]:
                    platform.record_decision(db, c, java, "Rejected",
                                             "Not a fit for this role" if not c.eligible else "Skills gap")
            java.apply_until = datetime.now() - timedelta(minutes=1)
            db.commit()
            platform.release_due_jobs(db)
            say("Java Developer (Freshers): deadline passed, results released.")

        say("\nDone. Log in with password", PASSWORD)
        say("  Recruiter: hr@acme.com")
        say("  Candidates: " + ", ".join(e for e, *_ in CANDIDATES))
    finally:
        db.close()




def seed_if_empty():
    """Deployed with HIREX_SEED_DEMO=1: fill an empty database once, on first start."""
    db = SessionLocal()
    try:
        empty = db.query(Recruiter).count() == 0
    finally:
        db.close()
    if empty:
        print("Empty database and HIREX_SEED_DEMO=1: loading demo data...")
        seed()

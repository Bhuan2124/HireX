"""The two-sided platform rules: deadlines, notifications, decisions shown to
candidates, duplicate applications and the candidate's "Why not me?" feedback.

Rules (agreed for HireX)
------------------------
- A candidate applies ONCE per job with ONE resume. No resubmission.
- Before the deadline a candidate sees only their match %, never their rank.
- When the deadline passes, the ranking is final: candidates see their rank,
  the interview zone and "Why not me?". There is no scheduler; the first page
  load after the deadline releases the results (release_due_jobs).
- A recruiter's decision (interview / selected / rejected) is shown to the
  candidate only after the deadline, so nobody learns their result early.
"""
from __future__ import annotations

import re
from datetime import datetime

from app.config import DUPLICATE_THRESHOLD
from app.models import Candidate, Job, Notification
from app.services import ranking, scoring

DECISION_STAGES = {"Shortlisted", "Interview", "Selected", "Rejected"}


# ------------------------------------------------------------------ notifications
def notify(db, user_type: str, user_id: int | None, text: str, link: str = ""):
    if user_id:
        db.add(Notification(user_type=user_type, user_id=user_id, text=text[:400], link=link))


def unread_count(db, user_type: str, user_id: int) -> int:
    return db.query(Notification).filter(Notification.user_type == user_type,
                                         Notification.user_id == user_id,
                                         Notification.is_read.is_(False)).count()


# ------------------------------------------------------------------ deadline
def release_due_jobs(db) -> int:
    """Release the results of every job whose deadline has passed. Cheap when
    there is nothing to do, so it is called on dashboard-type page loads."""
    due = (db.query(Job).filter(Job.apply_until.isnot(None),
                                Job.apply_until <= datetime.now(),
                                Job.results_released.is_(False)).all())
    for job in due:
        release_results(db, job)
    return len(due)


def release_results(db, job):
    cands = ranking.rerank(db, job.id)            # final ranking
    job.results_released = True
    for c in cands:
        if c.candidate_user_id:
            notify(db, "candidate", c.candidate_user_id,
                   f"Applications for {job.title} have closed. Your final rank and feedback are ready.",
                   f"/my/applications/{c.id}")
            if c.stage in DECISION_STAGES:     # decided before the deadline: tell them now
                notify(db, "candidate", c.candidate_user_id, decision_text(c, job),
                       f"/my/applications/{c.id}")
    notify(db, "recruiter", job.recruiter_id,
           f"Applications for {job.title} closed with {len(cands)} applicant"
           f"{'' if len(cands) == 1 else 's'}. The ranking is final.",
           f"/jobs/{job.id}/results")
    db.commit()


# ------------------------------------------------------------------ decisions
def decision_text(c: Candidate, job) -> str:
    return {"Shortlisted": f"Good news: you've been shortlisted for {job.title}.",
            "Interview": f"You've been called for interview for {job.title}"
                         + (f" on {c.interview_at}." if c.interview_at else "."),
            "Selected": f"Congratulations! You've been selected for {job.title}.",
            "Rejected": f"An update on your application for {job.title}."}.get(c.stage, "")


def record_decision(db, c: Candidate, job, stage: str, reason: str = "", note: str | None = None,
                    interview_at: str = ""):
    """Recruiter changes a stage. Notifies the candidate if results are out."""
    changed = c.stage != stage
    c.stage = stage
    if stage == "Rejected":
        c.decision_reason = reason or c.decision_reason or "Other"
    if stage == "Interview" and interview_at:
        c.interview_at = interview_at.strip()[:80]
    if note is not None:
        c.decision_note = note.strip()[:1000]
    if stage in DECISION_STAGES:
        c.decided_at = datetime.now()
    if changed and stage in DECISION_STAGES and job.results_released and c.candidate_user_id:
        notify(db, "candidate", c.candidate_user_id, decision_text(c, job), f"/my/applications/{c.id}")


def candidate_status(c: Candidate, job) -> dict:
    """What the candidate sees as the headline of their application."""
    if not job.results_released and not job.deadline_passed:
        return {"key": "review", "title": "Applied",
                "message": "Your application is in. Rankings and results are shown after the deadline"
                           + (f" ({job.apply_until:%d %b %Y, %I:%M %p})." if job.apply_until else ".")}
    s = c.stage
    if s == "Interview":
        return {"key": "good", "title": "Called for interview",
                "message": "The recruiter has invited you to the next round"
                           + (f" on {c.interview_at}." if c.interview_at else ". They will contact you with details.")}
    if s == "Selected":
        return {"key": "good", "title": "Selected",
                "message": "Congratulations, you have been selected for this role."}
    if s == "Shortlisted":
        return {"key": "good", "title": "Shortlisted",
                "message": "You are on the recruiter's shortlist. Watch this page for the next step."}
    if s == "Rejected":
        return {"key": "bad", "title": "Not selected",
                "message": f"Reason given by the recruiter: {c.decision_reason or 'not stated'}."}
    return {"key": "wait", "title": "Screening complete",
            "message": "Applications have closed and the ranking is final. "
                       "The recruiter hasn't made a decision on your application yet."}


# ------------------------------------------------------------------ duplicates
_WORD = re.compile(r"[a-z0-9]+")


def _shingles(text: str, n: int = 5) -> set[int]:
    w = _WORD.findall((text or "").lower())
    return {hash(tuple(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


def text_overlap(a: str, b: str) -> float:
    """Share of 5-word phrases two resumes have in common (Jaccard). Wording,
    not meaning: two genuine Java developers describe their work differently,
    a copied resume repeats the same sentences."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def flag_duplicate(db, cand: Candidate):
    """Compare a new application with the others for the same job."""
    others = db.query(Candidate).filter(Candidate.job_id == cand.job_id, Candidate.id != cand.id).all()
    best, best_sim = None, 0.0
    for o in others:
        if cand.email and o.email and cand.email.lower() == o.email.lower() \
                and o.candidate_user_id != cand.candidate_user_id:
            best, best_sim = o, 1.0                 # same email from a different account
            break
        sim = text_overlap(cand.resume_text, o.resume_text)
        if sim > best_sim:
            best, best_sim = o, sim
    if best is not None and best_sim >= DUPLICATE_THRESHOLD:
        cand.duplicate_of_id, cand.duplicate_similarity = best.id, round(best_sim, 3)
        if not best.duplicate_of_id:
            best.duplicate_of_id, best.duplicate_similarity = cand.id, round(best_sim, 3)
    db.commit()


# ------------------------------------------------------------------ "Why not me?"
def _practical_of(c: Candidate) -> dict:
    return {"internship": c.internship_score or 0.0, "projects": c.project_score or 0.0,
            "practical": c.practical_score or 0.0}


def overall_if_added(job, c: Candidate, skill: str) -> float:
    """The full score again, as if the resume showed this skill. Unlike a simple
    estimate this also lifts the 'no must-have skill' penalty when it applies."""
    sc = scoring.composite(job, set(c.skill_list) | {skill}, c.years_experience or 0.0,
                           c.education_level or 0, c.semantic_score or 0.0, _practical_of(c))
    return sc.overall


def components(job, c) -> list[dict]:
    w = job.weights
    rows = [("Skills", "skills", c.skill_score), ("Experience", "experience", c.experience_score),
            ("Practical work", "practical", c.practical_score), ("Education", "education", c.education_score),
            ("Overall fit", "semantic", c.semantic_score)]
    return [{"label": lbl, "key": k, "match": round((v or 0) * 100), "weight": round(w[k] * 100),
             "points": round((v or 0) * w[k] * 100, 1)} for lbl, k, v in rows if w[k] > 0]


def feedback(db, c: Candidate, job) -> dict:
    """Everything on the candidate's 'Why not me?' panel. Built only from the
    score arithmetic; other applicants are never named."""
    cands = (db.query(Candidate).filter(Candidate.job_id == job.id)
             .order_by(Candidate.rank.asc()).all())
    total = len(cands)
    eligible = [x for x in cands if x.eligible]
    zone = min(job.zone_size, len(eligible))
    in_zone = eligible[:zone]
    mine = components(job, c)

    # how far behind the interview zone, component by component (zone average)
    behind = []
    if in_zone and c not in in_zone:
        for row in mine:
            avg = sum((getattr(x, {"skills": "skill_score", "experience": "experience_score",
                                   "practical": "practical_score", "education": "education_score",
                                   "semantic": "semantic_score"}[row["key"]]) or 0)
                      for x in in_zone) / len(in_zone)
            gap = round((avg * job.weights[row["key"]] * 100) - row["points"], 1)
            if gap > 0.5:
                behind.append({"label": row["label"], "gap": gap, "zone_match": round(avg * 100),
                               "my_match": row["match"]})
        behind.sort(key=lambda r: -r["gap"])
    zone_avg = round(sum(x.overall_score for x in in_zone) / len(in_zone) * 100) if in_zone else 0
    cutoff = round(in_zone[-1].overall_score * 100) if in_zone else 0

    # missing skills: how much each would have added, and roughly which rank
    others = [x.overall_score for x in cands if x.id != c.id and x.eligible]
    gains = []
    for e in c.evidence:
        if e.status != "Missing":
            continue
        new = overall_if_added(job, c, e.skill)
        gain = round((new - c.overall_score) * 100, 1)
        if gain <= 0:
            continue
        est = 1 + sum(1 for s in others if s > new) if c.eligible else None
        gains.append({"skill": e.skill, "category": e.requirement_type, "gain": gain,
                      "rank": est, "related": e.related_skill})
    gains.sort(key=lambda g: (-g["gain"]))

    strong = [e for e in c.evidence if e.status != "Missing"]
    return {"rank": c.rank, "total": total, "zone": zone, "in_zone": c in in_zone,
            "places_away": max(0, c.rank - zone) if c.eligible else None,
            "zone_avg": zone_avg, "cutoff": cutoff, "mine": mine, "behind": behind[:4],
            "gains": gains[:5], "strong": strong, "practical": c.practical_detail,
            "filter_reason": "" if c.eligible else (c.filter_reason if not c.passes_filters else c.fit_reason)}

"""Database tables.

Two kinds of account use HireX:
  Recruiter     - posts jobs, reviews the ranking, decides who goes to interview
  CandidateUser - searches jobs, checks their match, applies, follows the result

A `Candidate` row is ONE APPLICATION: one resume for one job. It comes either
from a candidate applying on HireX (candidate_user_id is set) or from a recruiter
bulk-uploading resumes they received elsewhere (candidate_user_id is empty).
"""
import json
from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from app.config import SHORTLIST_MULTIPLIER
from app.database import Base


class Recruiter(Base):
    __tablename__ = "recruiters"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(190), unique=True, nullable=False, index=True)
    company = Column(String(150), default="")
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    jobs = relationship("Job", back_populates="recruiter", cascade="all, delete-orphan")


class CandidateUser(Base):
    """A job seeker's account (the candidate side of the platform)."""
    __tablename__ = "candidate_users"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(190), unique=True, nullable=False, index=True)
    phone = Column(String(40), default="")
    city = Column(String(80), default="")
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    applications = relationship("Candidate", back_populates="user")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    recruiter_id = Column(Integer, ForeignKey("recruiters.id"), nullable=False)
    title = Column(String(180), nullable=False)
    company = Column(String(150), default="")
    location = Column(String(120), default="")
    employment_type = Column(String(60), default="Full-time")
    category = Column(String(60), default="INFORMATION-TECHNOLOGY")
    description = Column(Text, default="")

    # platform settings
    experience_level = Column(String(20), default="Any")      # Fresher | Experienced | Any
    openings = Column(Integer, default=1)                     # how many people will be hired
    apply_until = Column(DateTime, nullable=True)             # application deadline
    shortlist_size = Column(Integer, default=0)               # 0 = openings x SHORTLIST_MULTIPLIER
    results_released = Column(Boolean, default=False)         # set once, right after the deadline

    # hard filters
    min_experience = Column(Float, default=0.0)
    min_education = Column(String(40), default="Any")

    # recruiter weights (sum to 1.0)
    w_skills = Column(Float, default=0.40)
    w_experience = Column(Float, default=0.25)
    w_practical = Column(Float, default=0.0)
    w_education = Column(Float, default=0.15)
    w_semantic = Column(Float, default=0.20)

    created_at = Column(DateTime, default=datetime.now)

    recruiter = relationship("Recruiter", back_populates="jobs")
    requirements = relationship("Requirement", back_populates="job",
                                cascade="all, delete-orphan")
    candidates = relationship("Candidate", back_populates="job",
                              cascade="all, delete-orphan")

    @property
    def weights(self):
        return {"skills": self.w_skills or 0.0, "experience": self.w_experience or 0.0,
                "practical": self.w_practical or 0.0,
                "education": self.w_education or 0.0, "semantic": self.w_semantic or 0.0}

    def skills_in(self, category):
        return [r.skill for r in self.requirements if r.category == category]

    # ---- deadline helpers (all times are the server's local time)
    @property
    def is_open(self) -> bool:
        return self.apply_until is None or datetime.now() < self.apply_until

    @property
    def deadline_passed(self) -> bool:
        return self.apply_until is not None and datetime.now() >= self.apply_until

    @property
    def days_left(self) -> int:
        if self.apply_until is None:
            return -1
        return max(0, (self.apply_until - datetime.now()).days)

    @property
    def zone_size(self) -> int:
        """How many top applicants are considered for interview."""
        return self.shortlist_size or max(1, (self.openings or 1) * SHORTLIST_MULTIPLIER)


class Requirement(Base):
    """One requirement: Must Have | Preferred | Nice to Have, with an importance."""
    __tablename__ = "requirements"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    skill = Column(String(120), nullable=False)
    category = Column(String(20), default="Must Have")
    importance = Column(Float, default=1.0)
    source = Column(String(20), default="extracted")     # extracted | manual

    job = relationship("Job", back_populates="requirements")


class Candidate(Base):
    """One application: one resume, parsed and scored against one job."""
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    candidate_user_id = Column(Integer, ForeignKey("candidate_users.id"), nullable=True)
    applied_via = Column(String(20), default="bulk")          # platform | bulk

    file_name = Column(String(255), default="")
    stored_path = Column(String(500), default="")

    # parsed profile
    name = Column(String(160), default="")
    email = Column(String(190), default="")
    phone = Column(String(60), default="")
    current_title = Column(String(180), default="")
    summary = Column(Text, default="")
    skills = Column(Text, default="")                 # "; " joined canonical skills
    education = Column(Text, default="")
    degree = Column(String(60), default="")
    education_level = Column(Integer, default=0)
    field_of_study = Column(String(120), default="")
    years_experience = Column(Float, default=0.0)
    experience_source = Column(String(45), default="")
    certifications = Column(Text, default="")
    projects = Column(Text, default="")
    work_experience = Column(Text, default="")
    predicted_category = Column(String(60), default="")
    resume_text = Column(Text, default="")

    # practical work (internships + projects), extracted once at parse time
    internship_months = Column(Integer, default=0)
    internship_count = Column(Integer, default=0)
    project_count = Column(Integer, default=0)
    practical_json = Column(Text, default="")          # what was found (see services/practical.py)
    practical_detail_json = Column(Text, default="")   # how it scored against this job

    # component scores (0-1)
    skill_score = Column(Float, default=0.0)
    experience_score = Column(Float, default=0.0)
    education_score = Column(Float, default=0.0)
    semantic_score = Column(Float, default=0.0)
    semantic_raw = Column(Float, default=0.0)          # raw SBERT cosine, for transparency
    semantic_key = Column(String(64), default="")      # hash of the job text it was computed for
    project_score = Column(Float, default=0.0)         # projects part of practical work
    internship_score = Column(Float, default=0.0)      # internships part of practical work
    practical_score = Column(Float, default=0.0)       # internships + projects combined
    overall_score = Column(Float, default=0.0)

    # model prediction (SVM / Random Forest trained on the HireX dataset)
    ml_score = Column(Float, default=0.0)
    ml_label = Column(Integer, default=0)

    passes_filters = Column(Boolean, default=True)
    filter_reason = Column(String(255), default="")
    fit_label = Column(String(30), default="")          # Strong / Good / Low / Weak match / Not eligible
    fit_reason = Column(String(255), default="")        # why "Not eligible"
    rank = Column(Integer, default=0)
    stage = Column(String(40), default="AI Screened")

    # recruiter decision, shown to the candidate after the deadline
    decision_reason = Column(String(80), default="")
    decision_note = Column(Text, default="")
    interview_at = Column(String(80), default="")
    decided_at = Column(DateTime, nullable=True)

    # integrity: near-copy of another application to the same job
    duplicate_of_id = Column(Integer, nullable=True)
    duplicate_similarity = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.now)

    job = relationship("Job", back_populates="candidates")
    user = relationship("CandidateUser", back_populates="applications")
    evidence = relationship("SkillEvidence", back_populates="candidate",
                            cascade="all, delete-orphan")

    @property
    def skill_list(self):
        return [s for s in (self.skills or "").split("; ") if s]

    @property
    def match_percent(self):
        return round(self.overall_score * 100)

    @property
    def eligible(self) -> bool:
        """Meets the hard filters AND is not 'Not eligible' (no must-have skill,
        different field). Only eligible applicants can be in the interview zone."""
        return bool(self.passes_filters) and self.fit_label != "Not eligible"

    @property
    def practical(self) -> dict:
        try:
            return json.loads(self.practical_json or "{}")
        except ValueError:
            return {}

    @property
    def practical_detail(self) -> dict:
        try:
            return json.loads(self.practical_detail_json or "{}")
        except ValueError:
            return {}


class MatchCheck(Base):
    """'Check my match' before applying. The resume is read and scored, but it is
    NOT an application until the candidate presses Apply. Kept so the same result
    is used for the application, without reading the file twice."""
    __tablename__ = "match_checks"
    id = Column(Integer, primary_key=True)
    candidate_user_id = Column(Integer, ForeignKey("candidate_users.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    file_name = Column(String(255), default="")
    stored_path = Column(String(500), default="")
    parsed_json = Column(Text, default="")             # parser output
    analysis_json = Column(Text, default="")           # scores, label, gaps
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)


class Notification(Base):
    """In-app notification for either kind of account (the bell icon)."""
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True)
    user_type = Column(String(12), nullable=False)      # recruiter | candidate
    user_id = Column(Integer, nullable=False, index=True)
    text = Column(String(400), nullable=False)
    link = Column(String(300), default="")
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)


class SkillEvidence(Base):
    """Why a candidate matched: one row per required / preferred skill."""
    __tablename__ = "skill_evidence"
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    skill = Column(String(120), nullable=False)
    requirement_type = Column(String(20), default="Must Have")
    status = Column(String(16), default="Missing")      # Strong | Partial | Missing
    mentions = Column(Integer, default=0)
    matched_surface = Column(String(255), default="")
    evidence = Column(Text, default="")
    related_skill = Column(String(120), default="")    # SBERT: closest skill the candidate has
    related_score = Column(Float, default=0.0)

    candidate = relationship("Candidate", back_populates="evidence")

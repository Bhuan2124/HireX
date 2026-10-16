"""Database tables: recruiters, jobs, requirements, candidates, skill evidence."""
from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from app.database import Base


class Recruiter(Base):
    __tablename__ = "recruiters"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(190), unique=True, nullable=False, index=True)
    company = Column(String(150), default="")
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("Job", back_populates="recruiter", cascade="all, delete-orphan")


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

    # hard filters
    min_experience = Column(Float, default=0.0)
    min_education = Column(String(40), default="Any")

    # recruiter weights (sum to 1.0)
    w_skills = Column(Float, default=0.40)
    w_experience = Column(Float, default=0.25)
    w_education = Column(Float, default=0.15)
    w_semantic = Column(Float, default=0.20)

    created_at = Column(DateTime, default=datetime.utcnow)

    recruiter = relationship("Recruiter", back_populates="jobs")
    requirements = relationship("Requirement", back_populates="job",
                                cascade="all, delete-orphan")
    candidates = relationship("Candidate", back_populates="job",
                              cascade="all, delete-orphan")

    @property
    def weights(self):
        return {"skills": self.w_skills, "experience": self.w_experience,
                "education": self.w_education, "semantic": self.w_semantic}

    def skills_in(self, category):
        return [r.skill for r in self.requirements if r.category == category]


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
    """One uploaded resume, parsed and scored against one job."""
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)

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

    # component scores (0-1)
    skill_score = Column(Float, default=0.0)
    experience_score = Column(Float, default=0.0)
    education_score = Column(Float, default=0.0)
    semantic_score = Column(Float, default=0.0)
    semantic_raw = Column(Float, default=0.0)          # raw SBERT cosine, for transparency
    semantic_key = Column(String(64), default="")      # hash of the job text it was computed for
    project_score = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)

    # model prediction (SVM / Random Forest trained on the ResumeX dataset)
    ml_score = Column(Float, default=0.0)
    ml_label = Column(Integer, default=0)

    passes_filters = Column(Boolean, default=True)
    filter_reason = Column(String(255), default="")
    rank = Column(Integer, default=0)
    stage = Column(String(40), default="AI Screened")
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="candidates")
    evidence = relationship("SkillEvidence", back_populates="candidate",
                            cascade="all, delete-orphan")

    @property
    def skill_list(self):
        return [s for s in (self.skills or "").split("; ") if s]

    @property
    def match_percent(self):
        return round(self.overall_score * 100)


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

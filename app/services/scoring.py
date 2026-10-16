"""The matching engine: component scores, hard filters, composite score and the
evidence that explains it.

Design rule: every number a recruiter sees comes out of this file, and the
explanation is generated from the same arithmetic that produced the rank. No
explanation is written after the fact.

The component definitions deliberately mirror the ResumeX dataset's scoring
config, so a model trained on the dataset sees the same features at run time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import EDUCATION_LEVELS, NO_MUST_HAVE_PENALTY, REQ_WEIGHT


# ------------------------------------------------------------------ components
def skill_component(candidate_skills: set[str], requirements: list) -> tuple[float, dict]:
    """Importance-weighted coverage of the job's requirements.

    requirements: objects with .skill, .category, .importance
    """
    if not requirements:
        return 0.0, {}
    total = sum(REQ_WEIGHT.get(r.category, 1.0) * r.importance for r in requirements)
    got = sum(REQ_WEIGHT.get(r.category, 1.0) * r.importance
              for r in requirements if r.skill in candidate_skills)
    detail = {
        "matched": [r.skill for r in requirements if r.skill in candidate_skills],
        "missing": [r.skill for r in requirements if r.skill not in candidate_skills],
        "must_total": sum(1 for r in requirements if r.category == "Must Have"),
        "must_matched": sum(1 for r in requirements
                            if r.category == "Must Have" and r.skill in candidate_skills),
    }
    return (got / total if total else 0.0), detail


def experience_component(years: float, required: float) -> float:
    if required <= 0:
        return 1.0
    return min(1.0, round(years / required, 4))


def education_component(level: int, required_name: str) -> float:
    need = EDUCATION_LEVELS.get(required_name, 0)
    if need == 0:
        return 1.0
    if level == 0:
        return 0.0                       # no degree could be established
    if level >= need:
        return 1.0
    return max(0.0, round(1.0 - 0.34 * (need - level), 4))


def transferable_skills(candidate_skills: set[str], requirements: list,
                        related: dict[str, set[str]]) -> list[str]:
    """Skills the candidate has that are adjacent to a missing requirement."""
    missing = {r.skill for r in requirements if r.skill not in candidate_skills}
    out = []
    for gap in missing:
        for near in related.get(gap, set()):
            if near in candidate_skills and near not in out:
                out.append(near)
    return out[:8]


# ------------------------------------------------------------------ hard filters
def check_filters(years: float, level: int, job) -> tuple[bool, str]:
    reasons = []
    if job.min_experience and years + 1e-9 < job.min_experience:
        reasons.append(f"needs {job.min_experience:g} yrs experience, resume shows {years:g}")
    need = EDUCATION_LEVELS.get(job.min_education, 0)
    if need and level and level < need:
        reasons.append(f"education below {job.min_education}")
    if need and level == 0:
        reasons.append("no degree found in resume")
    return (not reasons), "; ".join(reasons)


# ------------------------------------------------------------------ composite
@dataclass
class Score:
    skills: float = 0.0
    experience: float = 0.0
    education: float = 0.0
    semantic: float = 0.0
    project: float = 0.0
    overall: float = 0.0
    passes: bool = True
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def as_percent(self):
        return {k: round(v * 100) for k, v in
                dict(skills=self.skills, experience=self.experience,
                     education=self.education, semantic=self.semantic,
                     project=self.project, overall=self.overall).items()}


def composite(job, candidate_skills: set[str], years: float, level: int,
              semantic: float, project: float) -> Score:
    """Weighted candidate score. Weights come from the recruiter's sliders."""
    w = job.weights
    sk, detail = skill_component(candidate_skills, job.requirements)
    ex = experience_component(years, job.min_experience)
    ed = education_component(level, job.min_education)
    overall = (w["skills"] * sk + w["experience"] * ex +
               w["education"] * ed + w["semantic"] * semantic)

    # Documented rule: experience, education and semantic similarity must not add
    # up to a near-match score for someone who evidences none of the must-have
    # skills. The penalty is shown in the explanation, never applied silently.
    if detail.get("must_total") and detail.get("must_matched") == 0:
        overall *= NO_MUST_HAVE_PENALTY
        detail["penalty"] = (f"No must-have skill evidenced: score reduced to "
                             f"{int(NO_MUST_HAVE_PENALTY * 100)}% of the weighted total.")

    passes, reason = check_filters(years, level, job)
    return Score(skills=sk, experience=ex, education=ed, semantic=semantic,
                 project=project, overall=round(overall, 4),
                 passes=passes, reason=reason, detail=detail)


def uplift_if_added(job, candidate_skills: set[str], skill: str,
                    current: Score) -> float:
    """Counterfactual: how much would the overall score rise if the candidate
    had this one missing skill? Used for the 'what would help' line."""
    new_skills = set(candidate_skills) | {skill}
    sk, _ = skill_component(new_skills, job.requirements)
    return round((sk - current.skills) * job.weights["skills"] * 100, 1)


# ------------------------------------------------------------------ explanation
def explain(candidate, job, score: Score, evidence_rows: list) -> dict:
    """Assemble the 'Why selected?' payload straight from the score arithmetic."""
    w = job.weights
    contributions = [
        {"label": "Skills", "weight": round(w["skills"] * 100),
         "match": round(score.skills * 100), "points": round(score.skills * w["skills"] * 100, 1)},
        {"label": "Experience", "weight": round(w["experience"] * 100),
         "match": round(score.experience * 100),
         "points": round(score.experience * w["experience"] * 100, 1)},
        {"label": "Education", "weight": round(w["education"] * 100),
         "match": round(score.education * 100),
         "points": round(score.education * w["education"] * 100, 1)},
        {"label": "Overall fit", "weight": round(w["semantic"] * 100),
         "match": round(score.semantic * 100),
         "points": round(score.semantic * w["semantic"] * 100, 1)},
    ]
    contributions.sort(key=lambda c: -c["points"])
    strong = [e for e in evidence_rows if e.status == "Strong"]
    missing = [e for e in evidence_rows if e.status == "Missing"]

    top = contributions[0]
    weakest = min(contributions, key=lambda c: c["match"])
    sentences = [
        f"Overall {round(score.overall * 100)}% match, driven mainly by "
        f"{top['label'].lower()} ({top['points']} of 100 points).",
    ]
    if strong:
        sentences.append("Strong evidence for " +
                         ", ".join(e.skill for e in strong[:4]) +
                         (f" and {len(strong) - 4} more." if len(strong) > 4 else "."))
    if missing:
        sentences.append("No evidence found for " +
                         ", ".join(e.skill for e in missing[:4]) + ".")
    if weakest["match"] < 60:
        sentences.append(f"{weakest['label']} is the weakest component at {weakest['match']}%.")
    if score.detail.get("penalty"):
        sentences.append(score.detail["penalty"])
    if not score.passes:
        sentences.append(f"Hard requirement not met: {score.reason}.")

    return {"contributions": contributions,
            "explanation": " ".join(sentences),
            "strong": strong, "missing": missing}

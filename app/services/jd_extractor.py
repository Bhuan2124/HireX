"""Turn a pasted job description into editable, categorised requirements.

The extractor is rule-based and evidence-driven: a skill is only proposed if the
ontology matches it somewhere in the job description. Its category (Must Have /
Preferred / Nice to Have) comes from the sentence the skill appears in, using the
signal words recruiters actually write ("required", "preferred", "a plus").
"""
from __future__ import annotations

import re

from app.services import extractor as X
from app.services.ontology import ONTOLOGY

MUST_WORDS = re.compile(
    r"\b(must[- ]have|must|required|require[sd]?|essential|mandatory|minimum|"
    r"strong (?:experience|knowledge)|proficien|expert|should have)\b", re.I)
NICE_WORDS = re.compile(
    r"\b(nice[- ]to[- ]have|nice|bonus|a plus|advantage|desirable|good to have|"
    r"familiarity|exposure|awareness)\b", re.I)
PREF_WORDS = re.compile(r"\b(preferred|preferably|ideally|good to have|added advantage)\b", re.I)

EXP_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:to\s*\d{1,2}\s*)?(?:years|yrs)\b", re.I)
DEGREE_HINTS = [
    (5, r"\b(ph\.?d|doctorate)\b"),
    (4, r"\b(master|m\.?tech|m\.?s\b|mba|m\.?e\b|post[- ]graduate)\b"),
    (3, r"\b(bachelor|b\.?tech|b\.?e\b|b\.?sc|b\.?s\b|graduate degree|degree)\b"),
    (2, r"\b(diploma|associate)\b"),
    (1, r"\b(high school|12th|hsc)\b"),
]
LEVEL_NAME = {0: "Any", 1: "High School", 2: "Associate/Diploma",
              3: "Bachelor", 4: "Master", 5: "Doctorate"}


def _sentences(text: str):
    text = X.flat(text)
    # NB: do not split on ":" - "Preferred: Python, SQL" must stay one sentence,
    # otherwise the signal word is separated from the skills it governs.
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n+|\u2022", text) if s.strip()]


def categorise(sentence: str) -> str:
    """Which requirement bucket does a sentence imply?"""
    if NICE_WORDS.search(sentence):
        return "Nice to Have"
    if PREF_WORDS.search(sentence):
        return "Preferred"
    if MUST_WORDS.search(sentence):
        return "Must Have"
    return "Must Have"          # default: stated without hedging


def extract_requirements(job_description: str, job_title: str = "") -> list[dict]:
    """-> [{skill, category, importance, evidence}] ordered Must -> Preferred -> Nice."""
    found: dict[str, dict] = {}
    for sent in _sentences(job_description):
        cat = categorise(sent)
        for skill in X.normalized_skills(sent):
            prev = found.get(skill)
            rank = {"Must Have": 3, "Preferred": 2, "Nice to Have": 1}
            if prev is None or rank[cat] > rank[prev["category"]]:
                found[skill] = {"skill": skill, "category": cat,
                                "evidence": sent[:200],
                                "type": "Technical" if ONTOLOGY[skill][0] == "T" else "Soft"}
    # skills named in the job title are always must-haves
    for skill in X.normalized_skills(job_title):
        found.setdefault(skill, {"skill": skill, "category": "Must Have",
                                 "evidence": f"In job title: {job_title}",
                                 "type": "Technical" if ONTOLOGY[skill][0] == "T" else "Soft"})
        found[skill]["category"] = "Must Have"

    order = {"Must Have": 0, "Preferred": 1, "Nice to Have": 2}
    out = sorted(found.values(), key=lambda r: (order[r["category"]], r["skill"]))
    for r in out:
        r["importance"] = {"Must Have": 1.0, "Preferred": 0.6, "Nice to Have": 0.3}[r["category"]]
    return out


def extract_experience(job_description: str) -> float:
    m = EXP_RE.search(job_description)
    return float(m.group(1)) if m else 0.0


def extract_education(job_description: str) -> str:
    for level, pattern in DEGREE_HINTS:
        if re.search(pattern, job_description, re.I):
            return LEVEL_NAME[level]
    return "Any"


def summarise(job_description: str, job_title: str = "") -> dict:
    reqs = extract_requirements(job_description, job_title)
    return {
        "requirements": reqs,
        "min_experience": extract_experience(job_description),
        "min_education": extract_education(job_description),
        "counts": {c: sum(1 for r in reqs if r["category"] == c)
                   for c in ("Must Have", "Preferred", "Nice to Have")},
    }

"""Checks for internship / project scoring, the fresher weights and the fit labels.

Runs without the server, the database or Sentence-BERT:
    python tests/test_practical_scoring.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as N

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import WEIGHTS_BY_LEVEL                  # noqa: E402
from app.services import extractor as X, practical, scoring  # noqa: E402

RESUMES = ROOT / "tools" / "demo_resumes"
REQS = [N(skill=X.canonical_skill(s), category=c, importance=i) for s, c, i in
        [("Java", "Must Have", 1.0), ("Spring Boot", "Must Have", 1.0),
         ("SQL", "Preferred", 0.6), ("Docker", "Nice to Have", 0.3)]]


def found_in(name):
    text = X.clean_text((RESUMES / name).read_text(encoding="utf-8"))
    sec = X.split_sections(text)
    years = X.experience_years(text, scope=sec.get("experience") or None)
    return practical.extract(sec, text, years), text


def job(level):
    w = WEIGHTS_BY_LEVEL[level]
    return N(requirements=REQS, weights=w, experience_level=level, min_experience=0.0 if level == "Fresher" else 2.0,
             min_education="Bachelor", category="INFORMATION-TECHNOLOGY")


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


def main():
    ok = True
    ananya, _ = found_in("ananya_rao.txt")
    sneha, _ = found_in("sneha_kulkarni.txt")
    rahul, _ = found_in("rahul_menon.txt")
    vikram, _ = found_in("vikram_iyer.txt")

    ok &= check(len(ananya["internships"]) == 2, "Ananya: 2 internships found")
    ok &= check(ananya["internship_months"] == 7, f"Ananya: 7 internship months (got {ananya['internship_months']})")
    ok &= check([p["title"] for p in ananya["projects"]] ==
                ["Library Management System", "Movie Recommender", "Portfolio Website"], "Ananya: 3 project titles")
    ok &= check(len(sneha["internships"]) == 0 and len(sneha["projects"]) == 2, "Sneha: no internship, 2 projects")
    ok &= check(len(rahul["internships"]) == 1 and len(rahul["projects"]) == 2, "Rahul: internship inside Experience + numbered projects")
    ok &= check(vikram["internships"][0].get("kind") == "job", "Vikram: full-time job counted in place of an internship")

    sa, ss = practical.score(REQS, ananya), practical.score(REQS, sneha)
    ok &= check(sa["practical"] > ss["practical"] > 0, "internships + projects beat projects only, projects only > 0")
    ok &= check(ss["detail"]["rule"] == "projects_only", "projects-only rule applied for Sneha")

    # an extra weak project must never lower the projects score
    one = {"internships": [], "internship_months": 0, "projects": ananya["projects"][:1]}
    three = {"internships": [], "internship_months": 0, "projects": ananya["projects"]}
    ok &= check(practical.score(REQS, three)["projects"] >= practical.score(REQS, one)["projects"],
                "adding weaker projects never lowers the score")

    # fresher job: no experience penalty, practical work counts
    skills = {"Java", "Spring Boot", "SQL"}
    f = scoring.composite(job("Fresher"), skills, 0.0, 3, 0.5, sa)
    e = scoring.composite(job("Experienced"), skills, 0.0, 3, 0.5, sa)
    ok &= check(f.passes and not e.passes, "0 years passes a fresher job, fails a 2-year experienced job")
    ok &= check(f.overall > e.overall, "a strong fresher scores higher on a fresher job")

    # labels
    none = scoring.composite(job("Fresher"), set(), 4.0, 3, 0.05, None)
    ok &= check(scoring.fit_label(none)[0] == "Not eligible", "no must-have skill + low fit -> Not eligible")
    ok &= check(scoring.fit_label(f)[0] in ("Strong match", "Good match"), f"good fresher -> {scoring.fit_label(f)[0]}")

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Measure how accurately ResumeX reads resumes, against hand-checked answers.

    python tools/evaluate_extraction.py path/to/JavaDeveloper
    python tools/evaluate_extraction.py path/to/JavaDeveloper --verbose

tools/data/java_resumes_ground_truth.json holds the correct name, job title,
years of experience, highest degree and document type for 38 of the 72 scanned
Java-developer resumes, read by a person from the resume images. 30 of them
("tuning") were used while improving the parser; 8 ("holdout") were labelled
afterwards as a check on resumes the rules were not tuned on. Every field
ResumeX extracts for those files is compared with the answers:

    name    same words as the true name
    title   at least 75% of the true title's words, at most 2 extra words
    years   within 1 year, or 10% for long careers ("Present" = today)
    degree  same level (0 none, 1 high school, 2 associate, 3 bachelor, 4 master, 5 doctorate)
    java    the skill "Java" was found (resumes only)
    doc     resume / cover letter / job description classified correctly

Each field is scored only on the files where it applies (a cover letter has
no candidate name to find, a job posting no Java skill to detect).
The answers are tied to that dataset's file names. For another dataset, write a
JSON file of the same shape and pass it with --truth.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.parser import parse_resume  # noqa: E402


def words(s):
    w = re.sub(r"[^a-z ]", " ", (s or "").lower()).split()
    return [{"jr": "junior", "sr": "senior"}.get(x, x) for x in w]


def title_ok(want, got):
    if not want:
        return True
    w, g = set(words(want)) - {"resume"}, set(words(got))
    return len(w & g) / len(w) >= 0.75 and len(g - w) <= 2


def applies(truth, field):
    if field == "name":
        return bool(truth["name"])
    if field == "title":
        return bool(truth["title"])
    if field == "java":
        return truth["doc"] == "resume"
    return True


def check(truth, p):
    return {
        "name": set(words(truth["name"])) == set(words(p["name"])),
        "title": title_ok(truth["title"], p["current_title"]),
        "years": abs(p["years_experience"] - truth["years"]) <= max(1.0, 0.1 * truth["years"]),
        "degree": p["education_level"] == truth["degree"],
        "java": truth["doc"] != "resume" or "Java" in p["skills"].split("; "),
        "doc": p.get("doc_type", "resume") == truth["doc"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder holding the resume files")
    ap.add_argument("--truth", default=str(ROOT / "tools" / "data" / "java_resumes_ground_truth.json"))
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    folder = Path(args.folder)
    files = [f for f in truth if (folder / f).exists()]
    if not files:
        sys.exit(f"None of the {len(truth)} files in {args.truth} are in {folder}.")

    def parse(f):
        try:
            return f, parse_resume(str(folder / f), f)
        except Exception as exc:                                   # noqa: BLE001
            return f, {"error": str(exc)}

    print(f"Reading {len(files)} resumes (scans are OCR'd, this takes a minute)...")
    with ThreadPoolExecutor(4) as pool:
        parsed = dict(pool.map(parse, files))

    fields = ["name", "title", "years", "degree", "java", "doc"]
    for group in sorted({truth[f].get("set", "all") for f in files}, reverse=True):
        report(group, [f for f in files if truth[f].get("set", "all") == group],
               truth, parsed, fields, args.verbose)


def report(group, files, truth, parsed, fields, verbose):
    right = dict.fromkeys(fields, 0)
    total = dict.fromkeys(fields, 0)
    n = 0
    for f in files:
        p = parsed[f]
        if "error" in p:
            print(f"  ERROR {f}: {p['error']}")
            continue
        n += 1
        ok = {k: v for k, v in check(truth[f], p).items() if applies(truth[f], k)}
        for k, v in ok.items():
            right[k] += v
            total[k] += 1
        if verbose and not all(ok.values()):
            got = {"name": p["name"], "title": p["current_title"], "years": p["years_experience"],
                   "degree": p["education_level"], "java": "Java" in p["skills"], "doc": p.get("doc_type")}
            bad = ", ".join(f"{k}: expected {truth[f].get(k, True)!r}, got {got[k]!r}" for k in ok if not ok[k])
            print(f"  {f}: {bad}")

    print(f"\nAccuracy on {n} hand-checked resumes ({group} set)\n")
    for k in fields:
        print(f"  {k:7s} {right[k]:3d}/{total[k]:<3d} {100 * right[k] / max(1, total[k]):5.1f}%")


if __name__ == "__main__":
    main()

"""Practical work: internships and projects.

Why this exists: a fresher has no full-time job, so the Experience score is 0 for
every fresher. Without this module a student with two relevant internships and
three strong projects ranks the same as a student with nothing. Practical work
gives them credit for what they actually built and did.

Same rule as the rest of HireX: only what is written in the resume counts.
Nothing is guessed. A project with no description gets little credit, and an
internship without dates counts as a short one, not a long one.

Two steps:
  extract(sections, text)          -> what was found  (done once, when the resume is read)
  score(requirements, found, sim)  -> how well it fits ONE job (done per job)

How the score is calculated (all values 0-1)
--------------------------------------------
Internships (a candidate with a full-time job and no internship: the job counts here)
    duration   = min(1, total months / 6)            6+ months = full marks
    relevance  = job skills used in the internships  (importance-weighted, like Skills)
                 blended 70/30 with SBERT similarity to the job when SBERT is available
    internship = 0.35 x duration + 0.65 x relevance

Projects (each project)
    relevance  = job skills used in that project, blended 70/30 with SBERT
    depth      = 0.4 x description length (40+ words = full)
               + 0.3 x has a link / deployment (GitHub, live demo...)
               + 0.3 x has a measurable result ("92% accuracy", "500 users")
    project    = 0.75 x relevance + 0.25 x depth
    projects   = 0.60 x best project + 0.25 x 2nd best + 0.15 x 3rd best
                 (an extra weak project can only add points, never lower the score)

Practical work
    both present      = 0.5 x internships + 0.5 x projects
    only one present  = 0.8 x that one   (no internship? projects still earn up to 80%)
    neither           = 0
"""
from __future__ import annotations

import re

from app.config import (INTERNSHIP_FULL_MONTHS, INTERNSHIP_UNKNOWN_MONTHS,
                        PRACTICAL_SINGLE_SHARE, PROJECT_RANK_WEIGHTS, REQ_WEIGHT)
from app.services import extractor as X

INTERN_RE = re.compile(r"\bintern(?:ship)?s?\b", re.I)
LINK_RE = re.compile(r"github\.com|gitlab\.com|bitbucket\.org|https?://|www\.|\bdeployed\b|live demo|"
                     r"play ?store|app ?store|hosted on|vercel|netlify|heroku|streamlit\.app", re.I)
METRIC_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\b\d[\d,]*\s*\+?\s*k?\s*(?:users|downloads|requests|records|images|"
                       r"rows|students|customers|visitors|transactions|queries|ms\b|seconds|x faster)|"
                       r"\baccuracy\b[^.\n]{0,25}\d|"
                       # "1200 votes", "50 shops": a count (not a year) followed by a plural noun
                       r"\b(?!(?:19|20)\d{2}\b)\d[\d,]*\+?\s*k?\s+[a-z]{3,}s\b", re.I)
_NUMBERED = re.compile(r"^\(?\d{1,2}[.)]\s+|^(?:project|title)\s*\d*\s*[:\-]", re.I)
_META_LINE = re.compile(r"^(?:tech|technolog|tools|stack|skills|role|duration|link|github|environment|"
                        r"platform|languages?|team|guide|mentor|description)\b", re.I)
_BULLET = re.compile(r"^[•·\-\*▪●–>]")
_ACTION = {"built", "developed", "created", "implemented", "designed", "used", "deployed", "trained",
           "achieved", "integrated", "wrote", "worked", "led", "improved", "reduced", "added", "made",
           "applied", "performed", "analyzed", "analysed", "collected", "tested", "optimized", "optimised",
           "using", "with", "a", "an", "the", "this", "it", "and"}


# =========================================================== extraction
def _date_matches(text: str):
    """[(start, end)] character positions of every date range in text."""
    out = []
    for i, rx in enumerate(X.RANGE_RES):
        if i == X.LOOSE:
            continue
        for m in rx.finditer(text):
            if not any(m.start() < e and s < m.end() for s, e in out):
                out.append(m.span())
    return sorted(out)


def _months_union(spans) -> int:
    if not spans:
        return 0
    spans = sorted(spans)
    total, (cs, ce) = 0, spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + ce - cs


def _title(block: str) -> str:
    first = next((ln.strip() for ln in re.split(r"\n|\s{3,}", block) if ln.strip()), "")
    for rx in X.RANGE_RES[:4]:
        first = rx.sub("", first)
    return X.flat(first).strip(" ,|-:()")[:90]


def _entry(block: str) -> dict:
    spans = X.date_spans(block, allow_bare=True)
    return {"title": _title(block) or "Internship",
            "months": _months_union(spans) if spans else None,
            "spans": spans,
            "text": X.flat(block)[:700]}


def _split_dated_blocks(section: str) -> list[str]:
    """An Internships section with several entries: cut it at each date range
    (from the start of that line), so every block holds one internship."""
    dates = _date_matches(section)
    if len(dates) <= 1:
        return [section]
    cuts = [0]
    for s, e in dates[1:]:
        nl = section.rfind("\n", 0, s)
        cut = nl + 1 if nl != -1 and s - nl <= 150 else s
        # a line that is only the dates belongs to the title line above it
        eol = section.find("\n", e)
        rest = section[cut:s] + section[e:eol if eol != -1 else len(section)]
        if nl > 0 and len(re.findall(r"[A-Za-z]{3,}", rest)) <= 2:
            prev = section.rfind("\n", 0, nl)
            title_line = section[prev + 1:nl].strip()
            if title_line and len(title_line.split()) <= 12 and not title_line.endswith("."):
                cut = prev + 1
        if cut > cuts[-1] + 20:
            cuts.append(cut)
    cuts.append(len(section))
    return [section[a:b] for a, b in zip(cuts, cuts[1:]) if section[a:b].strip()]


def extract_internships(sections: dict, text: str) -> list[dict]:
    entries: list[dict] = []
    sec = sections.get("internships", "")
    if sec.strip():
        entries += [_entry(b) for b in _split_dated_blocks(sec)]

    # Internships listed inside Experience ("Software Intern, Infosys  Jun 2024 - Aug 2024").
    # With no Experience and no Internships section, look through the whole resume.
    scope = sections.get("experience", "") or ("" if sec.strip() else text)
    taken_until = -1
    for m in INTERN_RE.finditer(scope):
        if m.start() < taken_until:
            continue
        nl = scope.rfind("\n", 0, m.start())
        ls = nl + 1 if nl != -1 and m.start() - nl <= 150 else max(0, m.start() - 100)
        block = scope[ls:m.end() + 320]
        nxt = INTERN_RE.search(scope, m.end() + 40)        # stop before the next internship
        if nxt and nxt.start() < m.end() + 320:
            block = scope[ls:nxt.start()]
        # the dates sit near the title: look a little before and after it
        near = scope[max(0, ls - 60):m.end() + 160]
        e = _entry(block)
        if e["months"] is None:
            spans = X.date_spans(near, allow_bare=True)
            if spans:
                e["months"], e["spans"] = _months_union(spans[:1]), spans[:1]
        entries.append(e)
        taken_until = ls + len(block)
    return entries[:6]


def _split_projects(section: str) -> list[list[str]]:
    lines = [ln.strip() for ln in section.split("\n")]
    if sum(1 for ln in lines if ln) <= 2 and len(section) > 200:
        lines = X.items(section)                  # dataset layout: items separated by 2+ spaces
    blocks, cur, has_desc = [], [], False
    for ln in lines:
        if not ln:
            if cur and has_desc:
                blocks.append(cur)
                cur, has_desc = [], False
            continue
        words = ln.split()
        first = re.sub(r"[^a-z]", "", words[0].lower()) if words else ""
        looks_title = bool(_NUMBERED.match(ln)) or (
            len(words) <= 10 and not _BULLET.match(ln) and not ln.endswith(".")
            and not _META_LINE.match(ln) and first not in _ACTION and ln[:1].isupper())
        if cur and (_NUMBERED.match(ln) or (looks_title and has_desc)):
            blocks.append(cur)
            cur, has_desc = [ln], False
        else:
            if cur and not _META_LINE.match(ln):
                has_desc = True
            cur.append(ln)
    if cur:
        blocks.append(cur)
    return [b for b in blocks if len(" ".join(b).split()) >= 4][:10]


def extract_projects(sections: dict) -> list[dict]:
    sec = sections.get("projects", "")
    if not sec.strip():
        return []
    out = []
    for lines in _split_projects(sec):
        # title = first line; a one-line project is cut at its first " - " or ". "
        first = _NUMBERED.sub("", lines[0])
        title = first if len(lines) > 1 else re.split(r"\s[-–:|]\s|\.\s", first, maxsplit=1)[0]
        title = re.split(r"\s[-–|]\s", title, maxsplit=1)[0]
        words = title.split()
        if len(words) > 8:          # "Inventory tracker Built an app..." -> cut before the verb
            cut = next((i for i, w in enumerate(words[1:], 1) if w.lower() in _ACTION and w[:1].isupper()), 8)
            title = " ".join(words[:cut])
        out.append({"title": X.flat(title).strip(" :.")[:90],
                    "text": X.flat(" ".join(lines))[:900]})
    return out


def extract(sections: dict, text: str, years: float = 0.0) -> dict:
    """Everything practical-work scoring needs, stored with the application.

    years: full-time experience found by the parser. A candidate with a real job
    but no internship gets that job counted in the internship slot: a full-time
    job is at least as good evidence of practical work as an internship."""
    interns = extract_internships(sections, text)
    exp = sections.get("experience", "")
    if not interns and years > 0 and exp.strip():
        months = int(round(years * 12))
        interns = [{"title": "Work experience", "months": months, "spans": [(0, months)],
                    "text": X.flat(exp)[:700], "kind": "job"}]
    return {"internships": [{k: v for k, v in e.items() if k != "spans"} for e in interns],
            "internship_months": _months_union([s for e in interns for s in e["spans"]])
                                 + INTERNSHIP_UNKNOWN_MONTHS * sum(1 for e in interns if e["months"] is None),
            "projects": extract_projects(sections)}


# =========================================================== scoring
def _has_skill(text: str, skill: str, found: dict) -> bool:
    if skill in found or X.canonical_skill(skill) in found:
        return True
    return X.custom_skill_hits(text, skill) is not None


def skill_cover(text: str, requirements) -> tuple[float | None, list[str]]:
    """Importance-weighted share of the job's requirements used in this text."""
    if not requirements:
        return None, []
    found = X.normalized_skills(text or "")
    total = got = 0.0
    hits = []
    for r in requirements:
        w = REQ_WEIGHT.get(r.category, 1.0) * (r.importance or 1.0)
        total += w
        if _has_skill(text, r.skill, found):
            got += w
            hits.append(r.skill)
    return (got / total if total else None), hits


def _relevance(cover, sim) -> float:
    if cover is None and sim is None:
        return 0.0
    if cover is None:
        return sim
    if sim is None:
        return cover
    return 0.7 * cover + 0.3 * sim


def depth(text: str) -> tuple[float, dict]:
    words = len((text or "").split())
    marks = {"described": min(1.0, words / 40), "link": bool(LINK_RE.search(text or "")),
             "result": bool(METRIC_RE.search(text or ""))}
    return round(0.4 * marks["described"] + 0.3 * marks["link"] + 0.3 * marks["result"], 4), marks


def score(requirements, found: dict, sim_fn=None) -> dict:
    """Score practical work against one job.

    requirements: objects with .skill .category .importance (the job's requirements)
    found:        output of extract()
    sim_fn:       optional, list[str] -> list[float] calibrated SBERT similarity to the job
    """
    interns = found.get("internships") or []
    projects = found.get("projects") or []
    texts = [" ".join(e["text"] for e in interns)] if interns else []
    texts += [p["text"] for p in projects]
    sims = list(sim_fn(texts)) if (sim_fn and texts) else [None] * len(texts)

    # ---- internships
    i_score, i_detail = 0.0, {}
    if interns:
        months = found.get("internship_months", 0)
        duration = min(1.0, months / INTERNSHIP_FULL_MONTHS)
        cover, hits = skill_cover(texts[0], requirements)
        rel = _relevance(cover, sims[0])
        i_score = round(0.35 * duration + 0.65 * rel, 4)
        i_detail = {"count": len(interns), "months": months, "duration": round(duration, 3),
                    "is_job": any(e.get("kind") == "job" for e in interns),
                    "relevance": round(rel, 3), "skills_used": hits,
                    "undated": sum(1 for e in interns if e.get("months") is None),
                    "entries": [{"title": e["title"], "months": e.get("months")} for e in interns]}

    # ---- projects
    p_score, p_rows = 0.0, []
    off = 1 if interns else 0
    for k, p in enumerate(projects):
        cover, hits = skill_cover(p["text"], requirements)
        rel = _relevance(cover, sims[off + k])
        d, marks = depth(p["text"])
        p_rows.append({"title": p["title"], "score": round(0.75 * rel + 0.25 * d, 4),
                       "relevance": round(rel, 3), "depth": d, "skills_used": hits, **marks})
    if p_rows:
        # best project counts 60%, 2nd 25%, 3rd 15%: an extra project can only add points
        best = sorted(p_rows, key=lambda r: -r["score"])[:len(PROJECT_RANK_WEIGHTS)]
        p_score = round(sum(w * r["score"] for w, r in zip(PROJECT_RANK_WEIGHTS, best)), 4)

    # ---- combine
    if interns and p_rows:
        practical, rule = 0.5 * i_score + 0.5 * p_score, "both"
    elif p_rows:
        practical, rule = PRACTICAL_SINGLE_SHARE * p_score, "projects_only"
    elif interns:
        practical, rule = PRACTICAL_SINGLE_SHARE * i_score, "internships_only"
    else:
        practical, rule = 0.0, "none"

    # must-have skills not used in any internship or project
    used = set(i_detail.get("skills_used", [])) | {s for r in p_rows for s in r["skills_used"]}
    must_unused = [r.skill for r in (requirements or [])
                   if r.category == "Must Have" and r.skill not in used]

    return {"internship": i_score, "projects": p_score, "practical": round(practical, 4),
            "detail": {"rule": rule, "internships": i_detail, "projects": p_rows,
                       "must_unused": must_unused, "notes": notes(rule, i_detail, p_rows, must_unused)}}


def notes(rule, i_detail, p_rows, must_unused) -> list[str]:
    """Plain-language lines for the recruiter and for the candidate's 'Why not me?'."""
    out = []
    if rule == "none":
        out.append("No internships or projects were found in the resume.")
        return out
    if i_detail and i_detail.get("is_job"):
        out.append(f"No internship listed; {i_detail['months']} months of work experience are counted in its place.")
    elif i_detail:
        m = i_detail["months"]
        out.append(f"{i_detail['count']} internship{'s' if i_detail['count'] > 1 else ''}, "
                   f"about {m} month{'s' if m != 1 else ''} in total"
                   + (" (some without dates, counted as short)" if i_detail.get("undated") else "") + ".")
        if m < INTERNSHIP_FULL_MONTHS:
            out.append(f"Internships shorter than {INTERNSHIP_FULL_MONTHS} months get partial duration marks.")
    else:
        out.append(f"No internship found, so projects count for {int(PRACTICAL_SINGLE_SHARE * 100)}% "
                   "of practical work.")
    if p_rows:
        out.append(f"{len(p_rows)} project{'s' if len(p_rows) > 1 else ''} found. The best one counts 60%, "
                   "the 2nd 25% and the 3rd 15%, so more good projects means a higher score.")
        if not any(r["link"] for r in p_rows):
            out.append("No project has a GitHub or live link.")
        if not any(r["result"] for r in p_rows):
            out.append("No project states a measurable result (accuracy, users, speed...).")
    elif i_detail:
        out.append(f"No projects found, so internships count for {int(PRACTICAL_SINGLE_SHARE * 100)}% "
                   "of practical work.")
    if must_unused:
        out.append("Must-have skills not used in any internship or project: " + ", ".join(must_unused) + ".")
    return out

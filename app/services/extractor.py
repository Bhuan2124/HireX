"""Rule-based extraction of structured fields from resume text.
Deterministic: no ML, no randomness, no invented content.
Every extracted value is a substring (or a value computed from substrings) of the resume.
"""
import re, unicodedata
from app.services.ontology import ONTOLOGY

REF_DATE = (2020, 1)   # 'Current' anchor; corpus scraped ~2019-2020. Documented in methodology.

# ---------------- text normalisation ----------------
def clean_text(t):
    """Unicode-normalise but PRESERVE runs of spaces: this corpus uses 2+ spaces
    as its section-heading and list-item separator, so collapsing them loses layout."""
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\xa0", " ").replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\uff0d", "-").replace("\u2019", "'")
    t = t.replace("\t", "  ")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def flat(t):
    """Display form: collapse whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", t or "").strip()

def items(t):
    """Split a section body into list items using the corpus's 2+ space / newline / bullet layout."""
    parts = re.split(r"\s{2,}|\n|\u2022|\u00b7", t or "")
    return [flat(p) for p in parts if flat(p)]

# ---------------- section splitting ----------------
SECTION_WORDS = {
    "summary": ["summary", "professional summary", "career summary", "profile", "objective", "career objective", "professional profile", "executive summary"],
    "highlights": ["highlights", "core competencies", "key competencies", "areas of expertise", "core qualifications", "qualifications", "key skills"],
    "accomplishments": ["accomplishments", "achievements", "awards", "honors", "certifications", "licenses", "certifications and licenses", "affiliations", "professional affiliations"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history", "career history", "professional background"],
    "education": ["education", "education and training", "academic background", "educational qualifications", "academics"],
    "skills": ["skills", "technical skills", "skill set", "computer skills", "additional skills", "it skills"],
    "projects": ["projects", "key projects", "project experience", "academic projects", "project details"],
    "interests": ["interests", "activities", "hobbies", "languages", "personal details", "references", "publications", "military experience", "additional information"],
}
_ALL_HEADS = sorted({w for ws in SECTION_WORDS.values() for w in ws}, key=len, reverse=True)
_HEAD_RE = re.compile(r"(?:(?<=\s)|^)(" + "|".join(re.escape(w) for w in _ALL_HEADS) + r")(?=\s{2,}|\s*\n|\s*:)", re.I)
_HEAD_LOOKUP = {w: k for k, ws in SECTION_WORDS.items() for w in ws}

def split_sections(text):
    """Return {section_key: text}. Heading detection uses the double-space / newline
    layout that this corpus uses to mark section titles."""
    hits = [(m.start(), m.end(), _HEAD_LOOKUP[m.group(1).lower()]) for m in _HEAD_RE.finditer(text)]
    out, header_end = {}, hits[0][0] if hits else len(text)
    for i, (s, e, key) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[e:end].strip(" :\n")
        if len(body) > len(out.get(key, "")):
            out[key] = body
    out["_header"] = text[:header_end].strip()
    return out

# ---------------- job title ----------------
_TITLE_NOISE = re.compile(r"\b(summary|profile|objective|highlights|experience|resume|curriculum vitae)\b", re.I)

def job_title(sections, text):
    head = sections.get("_header", "")
    cand = [c for c in re.split(r"\n|\s{3,}", head) if c.strip()]
    cand = [c.strip(" -|,") for c in cand]
    for line in cand:
        line = _TITLE_NOISE.sub("", line).strip(" -–|,")
        line = flat(line)
        if 3 <= len(line) <= 70 and not re.search(r"\d{4}", line):
            return line
    m = re.search(r"\b(19|20)\d{2}\s+to\s+(current|present)\s+([A-Z][A-Za-z/&, .-]{3,60})", text)
    return flat(m.group(3)) if m else ""

# ---------------- summary ----------------
def summary(sections, limit=600):
    s = sections.get("summary", "")
    if not s:
        s = sections.get("highlights", "")
    s = flat(s)
    return s[:limit].rsplit(" ", 1)[0] if len(s) > limit else s

# ---------------- listed skills (verbatim) ----------------
def listed_skills(sections, cap=60):
    raw = sections.get("skills", "") or sections.get("highlights", "")
    if not raw:
        return []
    parts = []
    for it in items(raw):
        parts.extend(re.split(r"[,;|]+", it))
    out = []
    for p in parts:
        p = p.strip(" .-")
        if 2 <= len(p) <= 45 and not re.search(r"\d{4}", p) and p.lower() not in {w for ws in SECTION_WORDS.values() for w in ws}:
            if p not in out:
                out.append(p)
    return out[:cap]

# ---------------- normalised skills + evidence ----------------
# Token n-gram matcher: tokenise once, look up 1..4-token windows in a dict of
# surface form -> canonical skill. Deterministic and much faster than a mega-regex.
SEP = re.compile(r"[\-/._,()\[\]:;]")
TOK = re.compile(r"[a-z0-9+#&]+")
MAXN = 4

def _norm_surface(s):
    s = SEP.sub(" ", s.lower())
    return " ".join(TOK.findall(s))

def _build_lookup():
    lut = {}
    for canon, (typ, variants) in ONTOLOGY.items():
        for surface in [canon] + list(variants):
            key = _norm_surface(surface)
            if not key or len(key) < 2:
                continue
            lut.setdefault(key, (canon, surface))
    return lut

LOOKUP = _build_lookup()
MAXN = max(len(k.split()) for k in LOOKUP)

def _evidence(text, start, end, width=110):
    a, b = max(0, start - width), min(len(text), end + width)
    snip = flat(text[a:b])
    if a > 0:
        snip = "..." + snip
    if b < len(text):
        snip = snip + "..."
    return snip[:260]

def normalized_skills(text):
    """-> {canonical: {'surfaces': [...], 'evidence': str, 'count': n}}
    Longest n-gram wins at each position, so 'machine learning' is not also
    counted as a separate one-token match."""
    low = SEP.sub(" ", text.lower())
    toks = [(m.group(0), m.start(), m.end()) for m in TOK.finditer(low)]
    found, i, n = {}, 0, len(toks)
    while i < n:
        hit = None
        for L in range(min(MAXN, n - i), 0, -1):
            key = " ".join(t[0] for t in toks[i:i + L])
            if key in LOOKUP:
                hit = (L, LOOKUP[key])
                break
        if hit:
            L, (canon, surface) = hit
            rec = found.setdefault(canon, {"surfaces": [], "evidence": "", "count": 0})
            rec["count"] += 1
            if surface not in rec["surfaces"]:
                rec["surfaces"].append(surface)
            if not rec["evidence"]:
                rec["evidence"] = _evidence(text, toks[i][1], toks[i + L - 1][2])
            i += L
        else:
            i += 1
    return found

# ---------------- education ----------------
DEGREES = [
    (5, "Doctorate", [r"\bph\.?\s?d\b", r"\bdoctorate\b", r"\bdoctor of philosophy\b", r"\bm\.?d\.?\b(?!\w)", r"\bj\.?d\.?\b(?!\w)", r"\bed\.?d\b"]),
    (4, "Master", [r"\bmaster(?:'s)?\b", r"\bm\.?s\.?c?\b(?!\w)", r"\bm\.?b\.?a\b", r"\bm\.?tech\b", r"\bm\.?a\.?\b(?!\w)", r"\bm\.?com\b", r"\bmca\b", r"\bm\.?e\.?\b(?!\w)"]),
    (3, "Bachelor", [r"\bbachelor(?:'s)?\b", r"\bb\.?s\.?c?\b(?!\w)", r"\bb\.?tech\b", r"\bb\.?e\.?\b(?!\w)", r"\bb\.?a\.?\b(?!\w)", r"\bb\.?com\b", r"\bbba\b", r"\bbca\b", r"\bundergraduate degree\b"]),
    (2, "Associate/Diploma", [r"\bassociate(?:'s)? degree\b", r"\bassociate of (?:arts|science|applied)\b",
                              r"\ba\.?a\.?s\b", r"(?<!high school )(?<!school )\bdiploma\b", r"\bcertificate program\b"]),
    (1, "High School", [r"\bhigh school diploma\b", r"\bg\.?e\.?d\b", r"\bsecondary school\b", r"\bhigh school\b"]),
]
# Field of study: in this corpus the degree line reads "Bachelor of Science : Nursing   University Name",
# so the field ends at the next run of 2+ spaces. Captured only inside an Education section.
FIELD_RE = re.compile(
    r"\b(?:bachelor(?:'s)?|master(?:'s)?|associate(?:'s)?|doctorate|b\.?tech|m\.?tech|b\.?sc?|m\.?sc?|"
    r"b\.?a\.?|m\.?a\.?|mba|bba|bca|mca|phd|ph\.?d\.?)"
    r"(?:\s+(?:of|in))?\s*(?:degree)?\s*[:,]?\s*"
    r"(?:(?:of|in)\s+)?(?:science|arts|engineering|technology|business administration|applied science)?"
    r"\s*[:,]?\s*(?:(?:of|in)\s+)?"
    r"([A-Za-z][A-Za-z&/'. -]{3,50}?)(?=\s{2,}|\n|,\s|$)", re.I)

FIELD_STOP = re.compile(r"\b(university|college|institute|school|academy|city|state|current|present|"
                        r"n/?a|gpa|coursework|awarded|scholarship|high school)\b", re.I)

def education(sections, text):
    edu_txt = sections.get("education", "")
    scope = edu_txt if edu_txt else text
    strict_only = not edu_txt          # outside a real Education section, only full words count
    level, name = 0, ""
    for rank, label, pats in DEGREES:
        use = [p for p in pats if (not strict_only) or len(re.sub(r"\\W", "", p)) > 6]
        if any(re.search(p, scope, re.I) for p in use):
            level, name = rank, label
            break
    field = ""
    if edu_txt:                      # only trust a real Education section
        for m in FIELD_RE.finditer(edu_txt):
            f = flat(m.group(1)).strip(" ,.-:")
            f = re.split(r"\b(?:19|20)\d{2}\b", f)[0].strip(" ,.-:")
            if FIELD_STOP.search(f):
                f = FIELD_STOP.split(f)[0].strip(" ,.-:")
            if len(f) >= 4 and len(f.split()) <= 6 and not f.lower().startswith(("of ", "in ")):
                field = f[:60]
                break
    edu_clean = flat(edu_txt)[:400]
    return level, name, field, edu_clean

# ---------------- years of experience ----------------
MON = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
MONNUM = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
RANGE_RES = [
    re.compile(rf"\b({MON})[a-z]*\.?\s+((?:19|20)\d{{2}})\s*(?:to|-|–|until|through)\s*(?:({MON})[a-z]*\.?\s+((?:19|20)\d{{2}})|(current|present|till date|to date))", re.I),
    re.compile(r"\b(0?[1-9]|1[0-2])/((?:19|20)\d{2})\s*(?:to|-|–)\s*(?:(0?[1-9]|1[0-2])/((?:19|20)\d{2})|(current|present))", re.I),
    re.compile(r"\b((?:19|20)\d{2})\s*(?:to|-|–)\s*(?:((?:19|20)\d{2})|(current|present))\b", re.I),
]
STATED_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:plus\s*)?(?:years|yrs)\b[^.\n]{0,40}?\b(?:of\s+)?(?:experience|exp\b)", re.I)

def _months(y1, m1, y2, m2):
    return max(0, (y2 - y1) * 12 + (m2 - m1))

def experience_years(text, with_source=False, scope=None):
    """Union of dated employment intervals, in years (1 decimal).
    Falls back to an explicitly stated '<n>+ years of experience' claim.
    with_source=True also returns how the value was obtained."""
    # Prefer the Experience section: bare "1998 - 2002" ranges elsewhere are usually
    # education or award dates, not employment.
    src_text = scope if scope else text
    spans = []
    for i, rx in enumerate(RANGE_RES):
        if i == 2 and not scope:
            continue                      # bare-year ranges only trusted inside the experience section
        for m in rx.finditer(src_text):
            g = m.groups()
            try:
                if i == 0:
                    y1, m1 = int(g[1]), MONNUM.get(g[0][:3].lower(), 1)
                    if g[4]:
                        y2, m2 = REF_DATE
                    else:
                        y2, m2 = int(g[3]), MONNUM.get(g[2][:3].lower(), 1)
                elif i == 1:
                    y1, m1 = int(g[1]), int(g[0])
                    y2, m2 = (REF_DATE if g[4] else (int(g[3]), int(g[2])))
                else:
                    y1, m1 = int(g[0]), 1
                    y2, m2 = (REF_DATE if g[2] else (int(g[1]), 1))
            except (TypeError, ValueError):
                continue
            a, b = y1 * 12 + m1, y2 * 12 + m2
            if 1950 <= y1 <= 2026 and 1950 <= y2 <= 2026 and 0 < b - a <= 600:
                spans.append((a, b))
    total = 0
    if spans:
        spans.sort()
        cs, ce = spans[0]
        for s, e in spans[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                total += ce - cs
                cs, ce = s, e
        total += ce - cs
    yrs = round(total / 12.0, 1)
    src = ("dated_intervals_experience_section" if scope else "dated_intervals_full_text") if yrs > 0 else "not_found"
    if yrs == 0:
        m = STATED_RE.search(text)
        if m:
            yrs, src = float(m.group(1)), "stated_claim"
    yrs = min(yrs, 45.0)
    return (yrs, src) if with_source else yrs

# ---------------- certifications / projects / work experience ----------------
CERT_RE = re.compile(r"[^.\n•]{0,90}\b(certified|certification|certificate|licensed|license|credential)\b[^.\n•]{0,90}", re.I)

CERT_KW = re.compile(r"\b(certified|certification|certificate|licensure|licensed|license|credential|accreditation)\b", re.I)

def certifications(sections, text, cap=6):
    scope = " ".join(filter(None, [sections.get("accomplishments", ""), sections.get("education", ""),
                                   sections.get("highlights", ""), sections.get("_header", "")])) or text
    out = []
    for it in items(scope):
        for piece in re.split(r"(?<=[.;])\s+", it):
            piece = piece.strip(" ,.-")
            if CERT_KW.search(piece) and 8 <= len(piece) <= 140 and piece not in out:
                out.append(piece)
            if len(out) >= cap:
                return out
    return out

def projects(sections, limit=500):
    p = flat(sections.get("projects", ""))
    return p[:limit]

def work_experience(sections, limit=1500):
    w = flat(sections.get("experience", ""))
    return w[:limit]

INDUSTRY = {
    "INFORMATION-TECHNOLOGY": "Information Technology", "BUSINESS-DEVELOPMENT": "Business Services",
    "FINANCE": "Financial Services", "ACCOUNTANT": "Accounting & Audit", "BANKING": "Banking",
    "ENGINEERING": "Engineering & Manufacturing", "CONSTRUCTION": "Construction & Infrastructure",
    "AUTOMOBILE": "Automotive", "AVIATION": "Aviation & Aerospace", "AGRICULTURE": "Agriculture",
    "HEALTHCARE": "Healthcare", "FITNESS": "Health & Fitness", "CHEF": "Food & Hospitality",
    "HR": "Human Resources", "ADVOCATE": "Legal", "CONSULTANT": "Consulting",
    "PUBLIC-RELATIONS": "Media & Communications", "DIGITAL-MEDIA": "Digital Media",
    "DESIGNER": "Design & Creative", "ARTS": "Arts & Entertainment", "APPAREL": "Apparel & Fashion",
    "TEACHER": "Education", "SALES": "Sales & Retail", "BPO": "Business Process Outsourcing",
}

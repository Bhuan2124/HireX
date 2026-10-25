"""Rule-based extraction of structured fields from resume text.
Deterministic: no ML, no randomness, no invented content.
Every extracted value is a substring (or a value computed from substrings) of the resume.
"""
import os, re, unicodedata
from datetime import date
from app.services.ontology import IMPLIES, ONTOLOGY

# 'Current' / 'Present' anchor for experience ranges. Live uploads use today's
# date, otherwise "2022 - Present" would end in 2020 and be thrown away.
# To reproduce the dataset experiments (corpus scraped ~2019-2020) set
# RESUMEX_REF_DATE=2020-01.
def _ref_date():
    v = os.getenv("RESUMEX_REF_DATE", "")
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", v.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    t = date.today()
    return t.year, t.month

REF_DATE = _ref_date()
MAX_YEAR = REF_DATE[0] + 1

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

ROLE_WORDS = re.compile(r"\b(developer|engineer|programmer|architect|analyst|manager|consultant|"
                        r"designer|administrator|lead|intern|specialist|scientist|tester|officer|"
                        r"executive|associate|director|coordinator|technician|accountant)\b", re.I)

# Words that are never part of a person's name on a resume: section headings,
# contact labels, places, role words and common resume vocabulary.
NAME_BAD = set("""
resume curriculum vitae cv profile summary objective experience education skills skill contact details
personal info information philosophy expertise areas area my about me address city state zip code country
phone mobile tel email mail website web linkedin github portfolio page sample template references reference
history work employment professional career key qualifications qualification responsibilities strengths
achievements achievement languages language hobbies hobby interests interest united states street road avenue
drive lane technical core competencies competency certifications certification projects project java
developer developers senior junior lead jr sr intern entry level software engineer engineering full stack
react js the and with for years year dear sincerely hiring manager team problem solving management
development collaboration communication leadership analytical thinking attention detail adaptability
effective strong good excellent creative highlights accomplishments awards training get in touch hello
technologies technology systems solutions services inc ltd llc corp company corporation university college
school institute certified certifted certification programmer powered by enhancv qwikresume resumekraft resume worded cover letter job description
location remote overview declaration date place signature present current ongoing python sql
spring hibernate maven docker kubernetes aws git jira html css javascript angular agile scrum oracle
linux windows unix mysql struts servlets jsp rest api microservices
""".split())
_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-zA-Z'\u2019\-]+|[A-Z]\.?|(?:Jr|Sr)\.)$")
_VOWEL = re.compile(r"[aeiouyAEIOUY]")
_AFTER_NAME = re.compile(r"^(?:\(|\+|\d{3}[\s).-]|@|\||-|phone|tel|mobile|e-?mail|linkedin|website|address|senior|junior|lead|jr|sr|java|"
                         r"software|full|principal|staff|entry|associate)", re.I)

def _name_tokens(line):
    """Leading run of name-like words in a line ("Robert Smith Java Developer" ->
    ["Robert", "Smith"]); stops at the first word that cannot be part of a name."""
    out = []
    for tok in re.split(r"[\s|,•·]+", line.strip(" -|,:•")):
        if not tok:
            continue
        low = tok.lower().strip(".")
        if not _NAME_TOKEN.match(tok) or low in NAME_BAD:
            break
        out.append(tok)
        if len(out) == 4:
            break
    return out

def _fmt_name(tokens):
    return " ".join(t.title() if t.isupper() and len(t) > 2 else t for t in tokens)

def _valid_name(toks):
    long_ = [t for t in toks if len(t.strip(".")) >= 2]
    if len(long_) < 2 or len(toks) > 4:
        return False
    for t in long_:
        if not _VOWEL.search(t) or (t.isupper() and len(t) == 2):   # "Ss", "CA", "NY"
            return False
        if len(t) > 14 or _GARBLED.search(t):                        # OCR noise
            return False
    return True

_GARBLED = re.compile(r"(\w)\1\1|[aeiou]{4}|[^aeiouy\W\d_]{5}", re.I)

def _join_fragments(txt):
    """OCR splits letter-spaced names: "Ro bert S m ith" -> "Robert Smith",
    "ROBERT SM ITH" -> "ROBERT SMITH". A short capital fragment is glued to the
    capital word after it; a word starting in lower case to the word before it."""
    toks = txt.split()
    merged = []
    i = 0
    while i < len(toks):
        w = toks[i]
        while (w.isalpha() and w.isupper() and len(w) <= 2 and i + 1 < len(toks)
               and toks[i + 1].isalpha() and toks[i + 1].isupper()):
            i += 1
            w += toks[i]
        if (merged and w.isalpha() and w.isupper() and len(w) <= 2 and i == len(toks) - 1
                and merged[-1].isalpha() and merged[-1].isupper() and len(merged[-1]) > 2):
            merged[-1] += w                          # trailing piece: "WILL IS" -> "WILLIS"
        else:
            merged.append(w)
        i += 1
    out = []
    for w in merged:
        if out and w.isalpha() and out[-1].isalpha() and w[:1].islower():
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out)

def _name_from_line(txt):
    """Name at the start of a line, or "". The rest of the line may only be a
    role or contact details ("Robert Smith  Java Developer", "Janae Vandervort (555)")."""
    nm = _name_from_line_(txt)
    if not nm and re.search(r"\b\w{1,3}\b", txt):
        nm = _name_from_line_(_join_fragments(txt))
    if not nm and re.search(r"[a-z][A-Z]", txt):            # "LaceyWintheiser" (OCR lost the space)
        nm = _name_from_line_(re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", txt))
    if not nm and re.fullmatch(r"[A-Z]{6,20}", txt.strip()):  # "SIMONFORD" (OCR lost the space)
        parts = _word_split(txt.strip())
        if 2 <= len(parts) <= 3 and all(len(p) >= 2 for p in parts):
            nm = _name_from_line_(" ".join(parts))
    return nm


def _word_split(word):
    """Split a run of letters into English words (wordninja), if it is installed."""
    try:
        import wordninja
    except ImportError:
        return [word]
    return wordninja.split(word)

def _name_from_line_(txt):
    toks = _name_tokens(txt)
    words = txt.split()
    if not _valid_name(toks):
        return ""
    rest = " ".join(words[len(toks):]).strip(" ,|:-\u2022")
    if rest and not (_AFTER_NAME.match(rest) or ROLE_WORDS.match(rest)):
        return ""
    return _fmt_name(toks)

def person_name(lines, text=""):
    """The candidate's name. lines: page-1 [(text, size)] in reading order.

    1. Among the first lines of the page, the biggest one that reads as a name
       (2-4 capitalised words, optionally followed by a role or phone number).
       A name split over two lines ("CODY" / "MORRISON") is joined.
    2. Otherwise any clearly enlarged line elsewhere on the page (sidebar layouts).
    3. Otherwise the first plain 2-4 word line near the top of the text.
    """
    if lines:
        sizes = sorted(sz for _, sz in lines if sz > 0)
        median = sizes[len(sizes) // 2] if sizes else 0

        def at(i):
            txt, sz = lines[i]
            nm = _name_from_line(txt)
            if not nm:                                   # name split over two lines
                a_ = _name_tokens(txt)
                # the second half may be a few lines later when text beside the
                # name (a summary) was read in between; it has the same size
                for nxt_txt, nxt_sz in lines[i + 1:i + 7]:
                    if abs(nxt_sz - sz) > 0.25 * max(sz, 1):
                        continue
                    b_ = _name_tokens(nxt_txt)
                    if len(a_) == 1 == len(txt.split()) and 1 <= len(b_) == len(nxt_txt.split()) <= 2 \
                            and _valid_name(a_ + b_):
                        nm = _fmt_name(a_ + b_)
                    break
            return nm

        top = [(lines[i][1], -i, at(i)) for i in range(min(12, len(lines)))]
        top = [t for t in top if t[2]]
        best = None
        if top:
            # sizes within 15% of the biggest count as a tie: the earliest line wins
            # (word boxes with descenders, "Capital One", measure taller than "WILLIS")
            biggest = max(t[0] for t in top)
            best = max((t for t in top if t[0] >= 0.85 * biggest), key=lambda t: t[1])
        # Sidebar layouts: the name can come late in reading order, but it is
        # then clearly the biggest text (a quote's "- Napoleon Hill" is not).
        for i in sorted(range(len(lines)), key=lambda i: -lines[i][1])[:6]:
            sz = lines[i][1]
            if (median and sz < 1.3 * median) or (best and sz < 1.25 * best[0]):
                break
            nm = at(i)
            if nm:
                return nm
        if best:
            return best[2]
    for raw in (text or "").split("\n")[:12]:
        line = flat(raw)
        if not line or NAME_STOP_RE.search(line) or "@" in line:
            continue
        nm = _name_from_line(line)
        if nm:
            return nm
    return ""

NAME_STOP_RE = re.compile(r"\b(resume|curriculum|vitae|cv|profile|summary)\b", re.I)
_TITLE_DROP = re.compile(r"\b(resume|sample|template|cv|curriculum vitae)\b", re.I)
_TITLE_CUT = re.compile(r"\s*(?:\b(?:phone|tel|mobile|e-?mail|address|linkedin|website|www)\b|@|\(?\+?\d).*$", re.I)
_SENTENCE = re.compile(r"\b(with|years|experience|seeking|responsible|looking|by|who|i|am|my|at)\b", re.I)

_TITLE_MOD = re.compile(r"^(?:senior|sr|junior|jr|lead|principal|staff|chief|head|associate|assistant|"
                        r"entry|level|intermediate|mid|technical|full|stack|software|java|j2ee|backend|back-end|"
                        r"frontend|front-end|web|application|applications|core|contract|freelance|remote|"
                        r"cloud|devops|data|qa|test|it|systems?|solutions?|platform|enterprise|react|js|"
                        r"python|android|ios|mobile|net|\.net|/|&|and|of|-|\u2013)\.?$", re.I)

def _clean_title(line, exclude=""):
    """A job title from a header line: the role phrase ending in a role word,
    with junk around it removed ("sclestongraphicom SENIOR JAVA DEVELOPER",
    "Technical Lead / Senior Java Developer - Boehm, Houston, TX 10/2018")."""
    line = re.sub(r"(?<=\w)([(,])", r" \1", flat(line))
    if exclude and line.lower().startswith(exclude.lower()):
        line = line[len(exclude):]
    line = _TITLE_CUT.sub("", line)
    line = _TITLE_DROP.sub("", line)
    line = flat(line).strip(" -|,:.\u2022\u00b7")
    words = line.split()
    if not words or (_SENTENCE.search(line) and len(words) > 6):
        return ""
    roles = [i for i, w in enumerate(words) if ROLE_WORDS.fullmatch(w.strip(".,:;()"))]
    if not roles:
        return ""
    # stretch the phrase over role words and their modifiers around the first role word
    end = roles[0]
    while end + 1 < len(words) and (ROLE_WORDS.fullmatch(words[end + 1].strip(".,:;()"))
                                    or (_TITLE_MOD.match(words[end + 1]) and any(r > end + 1 for r in roles))):
        end += 1
    while end + 1 < len(words) and end + 1 in roles:
        end += 1
    start = roles[0]
    while start > 0 and _TITLE_MOD.match(words[start - 1]) and roles[0] - start < 4:
        start -= 1
    phrase = " ".join(words[start:end + 1]).strip(" -/&,.\u2013")
    if not (3 <= len(phrase) <= 60) or not phrase[:1].isupper():
        return ""
    return phrase.title() if phrase.isupper() else phrase

def job_title(sections, text, exclude="", lines=None):
    """Current / target job title: the first role line near the top of the page
    (normally right under the name), then any enlarged role line, then header
    lines, then the first role line of the Experience section. A one-word title
    ("Developer") is only used if nothing more specific is found."""
    pools = []
    if lines:
        if exclude:                     # the title normally sits right under the name
            key = exclude.lower().split()[0]
            for i, (t, _) in enumerate(lines):
                if key in t.lower():
                    pools.append([x for x, _ in lines[i:i + 4]])
                    break
        pools.append([t for t, _ in lines[:12]])
        sizes = sorted(sz for _, sz in lines if sz > 0)
        median = sizes[len(sizes) // 2] if sizes else 0
        pools.append([t for t, sz in sorted(lines, key=lambda l: -l[1])[:8] if sz >= 1.1 * median])
    pools.append([c for c in re.split(r"\n|\s{3,}", sections.get("_header", "")) if c.strip()][:15])
    pools.append((text or "").split("\n")[:15])
    pools.append(re.split(r"\n|\s{3,}", sections.get("experience", ""))[:12])
    fallback = ""
    for pool in pools:
        for line in pool:
            t = _clean_title(line, exclude)
            if t and len(t.split()) >= 2:
                return t
            fallback = fallback or t
    return fallback

# ---------------- document type ----------------
_COVER = [re.compile(p, re.I) for p in (
    r"\bdear\b", r"\bsincerely\b", r"\byours (?:truly|faithfully|sincerely)\b", r"\bhiring manager\b",
    r"\b(?:excited|writing|pleased|delighted) to apply\b", r"\bthank you for (?:your|taking the time)\b",
    r"\bto whom it may concern\b", r"\bcover letter\b", r"\blook forward to (?:meeting|hearing|discussing)\b")]
_JD = [re.compile(p, re.I) for p in (
    r"\bwe are (?:looking|seeking|hiring)\b", r"\bwe're (?:looking|hiring)\b", r"\bthe ideal candidate\b",
    r"\bjob description\b", r"\bminimum qualifications\b", r"\bpreferred qualifications\b",
    r"\bwhat you(?:'ll| will) do\b", r"\bperks\b", r"\bbenefits\b", r"\b(?:how to|to) apply\b",
    r"\bsend (?:us )?your (?:resume|cv)\b", r"\bequal opportunity employer\b", r"\bjoin our (?:team|company)\b",
    r"\bpaid (?:days|time) off\b", r"\bsalary range\b", r"\bjob (?:type|summary|requirements)\b")]

def document_type(text):
    """'resume', 'cover_letter' or 'job_description'. Bulk uploads scraped from
    the web often contain cover letters and job postings; those are flagged so
    they are not ranked as candidates."""
    t = text or ""
    dated = len(date_spans(t, allow_bare=True))
    cover = sum(1 for rx in _COVER if rx.search(t))
    jd = sum(1 for rx in _JD if rx.search(t))
    if cover >= 3 and dated < 2:
        return "cover_letter"
    if jd >= 3 and dated < 2:
        return "job_description"
    return "resume"

DOC_TYPE_LABEL = {"cover_letter": "a cover letter", "job_description": "a job description"}

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

# Single words that are also ordinary English or address words: only counted as
# the skill when a related technical word is close by ("Java, Spring, Hibernate"
# yes; "3043 Denise Spring, Houston" no).
CONTEXT_REQUIRED = {
    "spring": {"java", "boot", "mvc", "framework", "hibernate", "j2ee", "jee", "struts", "maven",
               "jpa", "rest", "restful", "api", "apis", "microservices", "microservice", "jsp",
               "servlets", "ioc", "aop", "cloud", "security", "data", "batch", "junit", "tomcat",
               "kafka", "sql", "jdbc", "mybatis", "ibatis", "angular", "react", "docker", "aws"},
}
CONTEXT_WINDOW = 6

def normalized_skills(text, implied=True):
    """-> {canonical: {'surfaces': [...], 'evidence': str, 'count': n}}
    Longest n-gram wins at each position, so 'machine learning' is not also
    counted as a separate one-token match.
    implied=True also credits the broader skill a specific one belongs to
    (Docker -> DevOps, Spring Boot -> Spring Framework), see ontology.IMPLIES."""
    low = SEP.sub(" ", (text or "").lower())
    toks = [(m.group(0), m.start(), m.end()) for m in TOK.finditer(low)]
    words = [t[0] for t in toks]
    found, i, n = {}, 0, len(toks)
    while i < n:
        hit = None
        for L in range(min(MAXN, n - i), 0, -1):
            key = " ".join(words[i:i + L])
            if key in LOOKUP:
                hit = (L, LOOKUP[key])
                break
        if hit and hit[0] == 1 and words[i] in CONTEXT_REQUIRED:
            near = set(words[max(0, i - CONTEXT_WINDOW):i] + words[i + 1:i + 1 + CONTEXT_WINDOW])
            if not near & CONTEXT_REQUIRED[words[i]]:
                hit = None
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
    if implied:
        for child in list(found):
            for parent in IMPLIES.get(child, []):
                rec = found.setdefault(parent, {"surfaces": [], "evidence": "", "count": 0})
                rec["count"] += found[child]["count"]
                for s in found[child]["surfaces"]:
                    if s not in rec["surfaces"]:
                        rec["surfaces"].append(s)
                if not rec["evidence"]:
                    rec["evidence"] = found[child]["evidence"]
    return found


def canonical_skill(name):
    """Map a skill typed by a recruiter ("java", "spring boot", "J2EE") onto the
    ontology's canonical name. Unknown skills are returned cleaned up as-is and
    are matched literally in resume text (see custom_skill_hits)."""
    key = _norm_surface(name or "")
    if key in LOOKUP:
        return LOOKUP[key][0]
    return flat(name)


def custom_skill_hits(text, skill):
    """Evidence dict (same shape as normalized_skills values) for a skill that is
    not in the ontology, matched as a whole phrase; None if absent."""
    key = _norm_surface(skill)
    if not key:
        return None
    rx = re.compile(r"(?<![a-z0-9+#])" + r"[\s\-/._,()]*".join(map(re.escape, key.split())) + r"(?![a-z0-9+#])")
    low = (text or "").lower()
    ms = list(rx.finditer(low))
    if not ms:
        return None
    return {"surfaces": [skill], "count": len(ms), "evidence": _evidence(text, ms[0].start(), ms[0].end())}

# ---------------- education ----------------
# Each pattern: (regex, strong). Full words are matched case-insensitively.
# Two-letter abbreviations (MS, MA, ME, BS, BA, BE) are also US states, pronouns
# and salutations ("Boston, MA", "Dear Ms.", "for me"), so they only count in
# capitals AND written as a degree: with dots (M.S.) or followed by "in"/"of"
# or a field (BS Computer Science). strong=False patterns are only trusted
# inside a real Education section.
_I = re.I
_FIELD_AFTER = (r"(?=\s*(?:\(|,|-|–)?\s*(?:in|of)\b|\s*(?:\(|,|-)?\s*(?:Computer|Information|"
                r"Software|Electrical|Electronics|Mechanical|Civil|Chemical|Data|Engineering|Science|"
                r"Sciences|Technology|Mathematics|Physics|Business|Commerce|CS|CSE|IT|ECE|EEE)\b)")
_ABBR = lambda a: (rf"(?<![A-Za-z]){a[0]}\.\s?{a[1]}\.|(?<![A-Za-z]){a[0]}\.\s?{a[1]}(?![A-Za-z])"
                   rf"|(?<![A-Za-z]){a}{_FIELD_AFTER}")
DEGREES = [
    (5, "Doctorate", [(re.compile(r"\bph\.?\s?d\b", _I), True), (re.compile(r"\bdoctorate\b", _I), True),
                      (re.compile(r"\bdoctor of philosophy\b", _I), True), (re.compile(r"\bM\.D\.(?!\w)"), False),
                      (re.compile(r"\bJ\.D\.(?!\w)"), False), (re.compile(r"\bed\.d\b", _I), False)]),
    (4, "Master", [(re.compile(r"(?<!scrum )(?<!data )\bmaster'?s?'?(?=\W|$)(?!\s+(?:data|card|class|"
                               r"plan|schedule|file|branch|node|thesis))", _I), True),
                   (re.compile(r"\bm\.?\s?tech\b", _I), True), (re.compile(r"\bm\.?\s?b\.?\s?a\b", _I), True),
                   (re.compile(r"\bMCA\b|\bmca\b"), True), (re.compile(r"\bM\.?\s?Com\b"), True),
                   (re.compile(r"\bM\.?\s?Sc\b"), True), (re.compile(r"\bM\.\s?E\.|\bM\.\s?A\."), False),
                   (re.compile(_ABBR("MS") + "|" + _ABBR("MA") + "|" + _ABBR("ME")), True)]),
    (3, "Bachelor", [(re.compile(r"\bbachelor'?s?'?(?=\W|$|[A-Z])"), True), (re.compile(r"\b[Bb]achelor'?s?'?(?=\W|$|[A-Z])|\bBACHELOR'?S?'?(?=\W|$)"), True), (re.compile(r"\bb\.?\s?tech\b", _I), True),
                     (re.compile(r"\bBBA\b|\bBCA\b"), True), (re.compile(r"\bB\.?\s?Com\b"), True),
                     (re.compile(r"\bB\.?\s?Sc\b"), True), (re.compile(r"\bundergraduate degree\b", _I), True),
                     (re.compile(r"\bB\.\s?E\.|\bB\.\s?A\."), False),
                     (re.compile(_ABBR("BS") + "|" + _ABBR("BA") + "|" + _ABBR("BE")), True)]),
    (2, "Associate/Diploma", [(re.compile(r"\bassociate'?s? degree\b", _I), True),
                              (re.compile(r"\bassociate of (?:arts|science|applied)", _I), True),
                              (re.compile(r"\bA\.?A\.?S\b"), False),
                              (re.compile(r"(?<!high school )(?<!school )\bdiploma\b", _I), True),
                              (re.compile(r"\bcertificate program\b", _I), False)]),
    (1, "High School", [(re.compile(r"\bhigh school diploma\b", _I), True), (re.compile(r"\bGED\b"), False),
                        (re.compile(r"\bsecondary school\b", _I), True), (re.compile(r"\bhigh school\b", _I), True)]),
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
        use = [rx for rx, strong in pats if strong or not strict_only]
        if any(rx.search(scope) for rx in use):
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
MON = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|dee|oet|0ct|rnay|rnar"   # last five: common OCR misreads
MONNUM = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONNUM.update({"dee": 12, "oet": 10, "0ct": 10, "rna": 5})
NOW = r"(current|presen[tl1]|pres[ae]nt|now|ongo[il1]ng|today|till date|to date|till now|date)"
DASH = r"\s*(?:to|-|–|until|through|till)\s*"
Y = r"((?:19|20)\d{2})"
# Each pattern yields named groups m1 y1 m2 y2 now. A missing month means January
# for a start and January for an end year, the conservative reading.
RANGE_RES = [
    # "Jan 2016 - Nov 2018", "September 2015 - Present", "Dec 2014 - 2020"
    re.compile(rf"(?<![A-Za-z])(?P<m1>{MON})[a-z]*\.?,?\s*'?{Y.replace('(', '(?P<y1>', 1)}{DASH}"
               rf"(?:(?:(?P<m2>{MON})[a-z]*\.?,?\s*)?{Y.replace('(', '(?P<y2>', 1)}|(?P<now>{NOW[1:-1]}))", re.I),
    # "05/2018 - PRESENT", "10/2012 - 02/2018", "06-2009 - 08-2012"
    re.compile(rf"\b(?P<m1>0?[1-9]|1[0-2])[/.-]{Y.replace('(', '(?P<y1>', 1)}{DASH}"
               rf"(?:(?P<m2>0?[1-9]|1[0-2])[/.-]{Y.replace('(', '(?P<y2>', 1)}|(?P<now>{NOW[1:-1]}))", re.I),
    # "2014-06 - 2016-05", "2016-06 - present", "2016/06 - 2018/01"
    re.compile(rf"\b{Y.replace('(', '(?P<y1>', 1)}[/.-](?P<m1>0[1-9]|1[0-2])\b{DASH}"
               rf"(?:{Y.replace('(', '(?P<y2>', 1)}[/.-](?P<m2>0[1-9]|1[0-2])\b|(?P<now>{NOW[1:-1]}))", re.I),
    # bare years "2018 - 2022", "2015 - Ongoing" (only inside an Experience section,
    # or anywhere for OCR'd scans, see experience_years)
    re.compile(rf"(?<![/.\-\d]){Y.replace('(', '(?P<y1>', 1)}(?![/.-](?:0[1-9]|1[0-2])\b){DASH}"
               rf"(?:{Y.replace('(', '(?P<y2>', 1)}(?![/.-](?:0[1-9]|1[0-2])\b)(?!\d)|(?P<now>{NOW[1:-1]})(?![a-z]))", re.I),
    # OCR of two-column templates puts the job title between the two dates:
    # "2016-08 - Senior Java Developer Present". Year-month dates only, used for scans.
    re.compile(rf"\b{Y.replace('(', '(?P<y1>', 1)}[/.-](?P<m1>0[1-9]|1[0-2])\b\s*-?[^\d]{{0,60}}?"
               rf"(?:{Y.replace('(', '(?P<y2>', 1)}[/.-](?P<m2>0[1-9]|1[0-2])\b|(?P<now>current|present|now|ongoing)\b)", re.I),
]
BARE = 3
LOOSE = 4
STATED_RE = re.compile(r"(?<![\d.])(\d{1,2})\s*\+?\s*(?:plus\s*)?(?:years|yrs)(?:of)?(?![a-z])[^.\n]{0,40}?(?:of\s*)?(?:experience|exp\b)", re.I)

def _months(y1, m1, y2, m2):
    return max(0, (y2 - y1) * 12 + (m2 - m1))

EDU_LINE = re.compile(r"(?i:\b(?:university|universit|college|school|institute|academy|bachelor\w*|master'?s?|"
                      r"b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|degree|gpa|cgpa|graduat\w*|"
                      r"class of|diploma|ph\.?\s?d|mba|bca|mca|ssc|hsc|cbse|education)\b)|\bB\.?\s?[SAE]\b|\bM\.?\s?[SAE]\b")
_WORD = re.compile(r"[A-Za-z]{3,}")
_DATE_WORDS = re.compile(rf"(?i)^(?:{MON}[a-z]*|{NOW[1:-1]}|to|until|through|till)$")

def _mon(v, default=1):
    if not v:
        return default
    if v.isdigit():
        return int(v)
    return MONNUM.get(v[:3].lower(), default)

def date_spans(text, allow_bare=True, skip_edu=False, loose=False):
    """[(start_month_index, end_month_index)] for every dated range in text."""
    lines = text.split("\n")
    if skip_edu:
        # drop ranges on a line that names a school/degree, or right under one
        keep = []
        for i, ln in enumerate(lines):
            prev = ""
            for back in lines[max(0, i - 2):i][::-1]:      # up to 2 lines up, not past a job title
                if ROLE_WORDS.search(back):
                    break
                prev += " " + back
            if ROLE_WORDS.search(ln):
                prev = ""
            # a line that is little more than a date, right under a school line,
            # belongs to that school ("University of X" / "2008 - Ongoing")
            only_date = len([w for w in _WORD.findall(ln) if not _DATE_WORDS.match(w)]) <= 3
            keep.append("" if EDU_LINE.search(ln) or (only_date and EDU_LINE.search(prev)) else ln)
        lines = keep
    src = "\n".join(lines)
    spans, covered = [], []
    for i, rx in enumerate(RANGE_RES):
        if (i == BARE and not allow_bare) or (i == LOOSE and not loose):
            continue
        for m in rx.finditer(src):
            if i in (BARE, LOOSE) and any(m.start() < e and s_ < m.end() for s_, e in covered):
                continue              # part of a "Sep 2015 - Present" already read with its month
            covered.append(m.span())
            g = m.groupdict()
            try:
                y1, m1 = int(g["y1"]), _mon(g.get("m1"))
                if g.get("now"):
                    y2, m2 = REF_DATE
                else:
                    y2, m2 = int(g["y2"]), _mon(g.get("m2"))
            except (TypeError, ValueError):
                continue
            a, b = y1 * 12 + m1, y2 * 12 + m2
            if 1950 <= y1 <= MAX_YEAR and 1950 <= y2 <= MAX_YEAR and 0 < b - a <= 600 \
                    and b <= REF_DATE[0] * 12 + REF_DATE[1] + 1:
                spans.append((a, b))
    return spans

def experience_years(text, with_source=False, scope=None, bare_years_anywhere=False):
    """Union of dated employment intervals, in years (1 decimal).
    Falls back to an explicitly stated '<n>+ years of experience' claim.
    with_source=True also returns how the value was obtained.

    bare_years_anywhere=True (used for OCR'd scans, whose two-column layout
    scrambles section boundaries) also accepts "2018 - 2022" ranges outside an
    Experience section, but ignores ranges on or right under a school / degree line."""
    # Prefer the Experience section: bare "1998 - 2002" ranges elsewhere are usually
    # education or award dates, not employment.
    if scope:
        spans = date_spans(scope, allow_bare=True)
    else:
        spans = date_spans(text, allow_bare=bare_years_anywhere, skip_edu=True, loose=bare_years_anywhere)
    total = 0
    if spans:
        spans.sort()
        cs, ce = spans[0]
        for s_, e in spans[1:]:
            if s_ <= ce:
                ce = max(ce, e)
            else:
                total += ce - cs
                cs, ce = s_, e
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

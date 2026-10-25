"""Shared Jinja2 template environment plus a few display filters."""
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def pct(value):
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "0%"


def years(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v:g} yr" if v == 1 else f"{v:g} yrs"


def highlight(evidence, surfaces):
    """Wrap the matched skill wording inside a resume sentence in <mark>, so the
    recruiter sees exactly which words the match was made on."""
    import html
    import re
    from markupsafe import Markup

    text = html.escape(evidence or "")
    terms = sorted({t.strip() for t in (surfaces or "").split(";") if t.strip()},
                   key=len, reverse=True)
    for term in terms:
        pattern = r"(?<![A-Za-z0-9])(" + r"[\s\-/]{1,3}".join(
            re.escape(w) for w in html.escape(term).split()) + r")(?![A-Za-z0-9])"
        text = re.sub(pattern, r"<mark>\1</mark>", text, flags=re.I)
    return Markup(text)


def band(value):
    """Match band used for colour and wording."""
    v = float(value or 0)
    return "strong" if v >= 0.70 else ("fair" if v >= 0.50 else "weak")


ACRONYMS = {"HR", "BPO", "IT"}


def field(code):
    """INFORMATION-TECHNOLOGY -> Information Technology, keeping HR and BPO as acronyms."""
    words = (code or "").replace("-", " ").split()
    return " ".join(w if w in ACRONYMS else w.capitalize() for w in words)


templates.env.filters["field"] = field
templates.env.filters["pct"] = pct
templates.env.filters["years"] = years
templates.env.filters["highlight"] = highlight
templates.env.filters["band"] = band

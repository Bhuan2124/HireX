"""Shared Jinja2 template environment plus a few display filters."""
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR

class _Templates(Jinja2Templates):
    """Accepts both ways of calling TemplateResponse.

    The routers use the older form TemplateResponse("page.html", {"request": ...}).
    Newer Starlette versions (pulled in by newer FastAPI, e.g. on Hugging Face)
    only accept TemplateResponse(request, "page.html", {...}); this converts.
    """

    def TemplateResponse(self, *args, **kwargs):  # noqa: N802 (Starlette's name)
        if args and isinstance(args[0], str):
            name, context, *rest = args
            context = dict(context or {})
            return super().TemplateResponse(context.get("request"), name, context, *rest, **kwargs)
        return super().TemplateResponse(*args, **kwargs)


templates = _Templates(directory=str(BASE_DIR / "templates"))


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


def unread(request):
    """Unread notification count for the bell in the header."""
    kind = "recruiter" if request.session.get("recruiter_id") else (
        "candidate" if request.session.get("candidate_id") else None)
    if not kind:
        return 0
    from app.database import SessionLocal
    from app.services.platform import unread_count
    db = SessionLocal()
    try:
        return unread_count(db, kind, request.session.get(f"{kind}_id"))
    finally:
        db.close()


def when(value):
    """20 Nov 2026, 5:00 PM"""
    if not value:
        return "-"
    return value.strftime("%d %b %Y, %I:%M %p").replace(" 0", " ")


def ordinal(n):
    n = int(n or 0)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


templates.env.globals["unread"] = unread
templates.env.filters["when"] = when
templates.env.filters["ordinal"] = ordinal

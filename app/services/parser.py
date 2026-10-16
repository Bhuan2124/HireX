"""Resume file parsing: PDF / DOCX / TXT -> raw text -> structured profile.

PDF text is read with PyMuPDF. Characters that are invisible (white, tiny or
zero-size) are dropped, because hidden text is the standard way applicants try
to manipulate automated screeners.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.services import extractor as X
from app.services.ontology import ONTOLOGY

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4})")
NAME_STOP = re.compile(r"\b(resume|curriculum|vitae|cv|profile|summary)\b", re.I)


# --------------------------------------------------------------- file readers
def read_pdf(path: str) -> tuple[str, list[str]]:
    """Return (visible_text, hidden_text_flags). Requires PyMuPDF."""
    import fitz                                    # PyMuPDF

    visible, hidden = [], []
    doc = fitz.open(path)
    for page in doc:
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if not txt.strip():
                        continue
                    size = span.get("size", 10)
                    colour = span.get("color", 0)          # 0 = black, 16777215 = white
                    if size < 4 or colour >= 16250000:
                        hidden.append(txt.strip())
                    else:
                        visible.append(txt)
                visible.append("  ")
            visible.append("\n")
    doc.close()
    return "".join(visible), hidden


def read_docx(path: str) -> tuple[str, list[str]]:
    import docx

    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append("  ".join(c.text for c in row.cells))
    return "\n".join(parts), []


def read_txt(path: str) -> tuple[str, list[str]]:
    return Path(path).read_text(encoding="utf-8", errors="ignore"), []


def read_any(path: str) -> tuple[str, list[str]]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".txt":
        return read_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


# --------------------------------------------------------------- contact details
def contact_details(text: str) -> tuple[str, str, str]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    name = ""
    for line in text.split("\n")[:8]:
        line = X.flat(line).strip(" -|,")
        if not line or NAME_STOP.search(line) or EMAIL_RE.search(line):
            continue
        words = line.split()
        if 1 < len(words) <= 4 and not re.search(r"\d", line) and len(line) <= 45:
            name = line.title() if line.isupper() else line
            break
    return name, (email.group(0) if email else ""), (phone.group(0) if phone else "")


# --------------------------------------------------------------- full profile
def parse_resume(path: str, file_name: str = "") -> dict:
    """Parse one resume file into the structured profile the matcher needs."""
    raw, hidden = read_any(path)
    text = X.clean_text(raw)
    sections = X.split_sections(text)

    name, email, phone = contact_details(text)
    level, degree, field, edu_text = X.education(sections, text)
    exp_section = sections.get("experience", "")
    years, source = X.experience_years(text, True, scope=exp_section or None)
    skills = X.normalized_skills(text)

    if not name and file_name:
        name = Path(file_name).stem.replace("_", " ").replace("-", " ").title()

    return {
        "file_name": file_name or Path(path).name,
        "name": name or "Candidate",
        "email": email,
        "phone": phone,
        "current_title": X.job_title(sections, text),
        "summary": X.summary(sections),
        "skills_detail": skills,                       # canonical -> evidence dict
        "skills": "; ".join(sorted(skills)),
        "technical_skills": "; ".join(sorted(s for s in skills if ONTOLOGY[s][0] == "T")),
        "soft_skills": "; ".join(sorted(s for s in skills if ONTOLOGY[s][0] == "S")),
        "education": edu_text,
        "degree": degree,
        "education_level": level,
        "field_of_study": field,
        "years_experience": years,
        "experience_source": source,
        "certifications": " | ".join(X.certifications(sections, text)),
        "projects": X.projects(sections),
        "work_experience": X.work_experience(sections, 1500),
        "resume_text": X.flat(text),
        "hidden_text": hidden,                         # integrity flag for the UI
        "sections": sorted(k for k in sections if k != "_header"),
    }

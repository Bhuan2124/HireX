"""Resume file parsing: PDF / DOCX / TXT / image -> raw text -> structured profile.

PDF text is read with PyMuPDF. Characters that are invisible (white, tiny or
zero-size) are dropped, because hidden text is the standard way applicants try
to manipulate automated screeners. Scanned pages (an image with no text layer)
and image files are read with Tesseract OCR.

Besides the text, each reader returns the first page's lines with their text
size. The candidate's name is almost always the biggest text on a resume and
the job title the next biggest, which is far more reliable than guessing from
line order (two-column layouts scramble the order, especially after OCR).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import IMAGE_EXTENSIONS, OCR_MIN_PAGE_CHARS
from app.services import extractor as X
from app.services import ocr
from app.services.ontology import ONTOLOGY

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4})")
NAME_STOP = re.compile(r"\b(resume|curriculum|vitae|cv|profile|summary)\b", re.I)


@dataclass
class Document:
    text: str
    hidden: list = field(default_factory=list)      # invisible-text integrity flags
    used_ocr: bool = False
    lines: list = field(default_factory=list)       # page 1: [(line text, text size)]


# --------------------------------------------------------------- file readers
def read_pdf(path: str) -> Document:
    """Requires PyMuPDF. A page with (almost) no text layer but an image on it
    is a scan: it is read with OCR instead (see app/services/ocr.py)."""
    import fitz                                    # PyMuPDF

    pages_out, hidden, first_lines = [], [], []
    ocr_error, used_ocr = "", False
    doc = fitz.open(path)
    try:
        for pno, page in enumerate(doc):
            visible, page_lines = [], []
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    line_txt, line_size = [], 0.0
                    for span in line.get("spans", []):
                        txt = span.get("text", "")
                        if not txt.strip():
                            continue
                        size = span.get("size", 10)
                        colour = span.get("color", 0)      # 0 = black, 16777215 = white
                        if size < 4 or colour >= 16250000:
                            hidden.append(txt.strip())
                        else:
                            visible.append(txt)
                            line_txt.append(txt)
                            line_size = max(line_size, float(size))
                    visible.append("  ")
                    if line_txt:
                        page_lines.append(("".join(line_txt).strip(), line_size))
                visible.append("\n")
            page_text = "".join(visible)

            if len(page_text.strip()) < OCR_MIN_PAGE_CHARS and page.get_images():
                try:
                    page_text, page_lines = ocr.pdf_page_to_layout(page)
                    used_ocr = True
                except ocr.OCRUnavailable as exc:
                    ocr_error = str(exc)
            pages_out.append(page_text)
            if pno == 0:
                first_lines = page_lines
    finally:
        doc.close()

    text = "\n".join(pages_out)
    if not text.strip() and ocr_error:
        raise ValueError("scanned PDF with no text layer, and OCR is not available. " + ocr_error)
    return Document(text, hidden, used_ocr, first_lines)


def read_image(path: str) -> Document:
    try:
        text, lines = ocr.image_file_to_layout(path)
    except ocr.OCRUnavailable as exc:
        raise ValueError("image resume needs OCR, which is not available. " + str(exc)) from exc
    return Document(text, [], True, lines)


def read_docx(path: str) -> Document:
    import docx

    d = docx.Document(path)
    parts, lines = [], []
    for p in d.paragraphs:
        parts.append(p.text)
        if p.text.strip() and len(lines) < 40:
            sizes = [r.font.size.pt for r in p.runs if r.font is not None and r.font.size]
            lines.append((p.text.strip(), max(sizes) if sizes else 0.0))
    for table in d.tables:
        for row in table.rows:
            parts.append("  ".join(c.text for c in row.cells))
    return Document("\n".join(parts), [], False, lines)


def read_txt(path: str) -> Document:
    return Document(Path(path).read_text(encoding="utf-8", errors="ignore"))


def read_any(path: str) -> Document:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".txt":
        return read_txt(path)
    if ext in IMAGE_EXTENSIONS:
        return read_image(path)
    raise ValueError(f"Unsupported file type: {ext}")


# --------------------------------------------------------------- contact details
def contact_details(text: str, lines: list | None = None) -> tuple[str, str, str]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    name = X.person_name(lines or [], text)
    return name, (email.group(0) if email else ""), (phone.group(0) if phone else "")


# --------------------------------------------------------------- full profile
def parse_resume(path: str, file_name: str = "") -> dict:
    """Parse one resume file into the structured profile the matcher needs."""
    doc = read_any(path)
    text = X.clean_text(doc.text)
    if len(text) < 40:
        # Scoring an empty resume gives every candidate the same meaningless
        # score (e.g. 20%), so report the file instead of ranking it.
        raise ValueError("no readable text found in the file")
    sections = X.split_sections(text)
    lines = [(X.clean_text(t), s) for t, s in doc.lines]

    name, email, phone = contact_details(text, lines)
    level, degree, field_, edu_text = X.education(sections, text)
    exp_section = sections.get("experience", "")
    years, source = X.experience_years(text, True, scope=exp_section or None)
    if doc.used_ocr or not exp_section:
        # OCR reads two-column layouts line by line across both columns, so the
        # Experience section boundaries are unreliable. Also count dated ranges
        # anywhere in the text (school / degree lines excluded) and keep the larger.
        y2, s2 = X.experience_years(text, True, bare_years_anywhere=doc.used_ocr)
        if y2 > years:
            years, source = y2, s2
    skills = X.normalized_skills(text)

    if not name:
        # fall back to the file name so the recruiter can still tell files apart
        name = Path(file_name or path).stem.replace("_", " ").replace("-", " ").strip() or "Candidate"

    return {
        "file_name": file_name or Path(path).name,
        "name": name,
        "email": email,
        "phone": phone,
        "current_title": X.job_title(sections, text, exclude=name, lines=lines),
        "summary": X.summary(sections),
        "skills_detail": skills,                       # canonical -> evidence dict
        "skills": "; ".join(sorted(skills)),
        "technical_skills": "; ".join(sorted(s for s in skills if ONTOLOGY[s][0] == "T")),
        "soft_skills": "; ".join(sorted(s for s in skills if ONTOLOGY[s][0] == "S")),
        "education": edu_text,
        "degree": degree,
        "education_level": level,
        "field_of_study": field_,
        "years_experience": years,
        "experience_source": source,
        "certifications": " | ".join(X.certifications(sections, text)),
        "projects": X.projects(sections),
        "work_experience": X.work_experience(sections, 1500),
        "resume_text": X.flat(text),
        "hidden_text": doc.hidden,                     # integrity flag for the UI
        "used_ocr": doc.used_ocr,
        "doc_type": X.document_type(text),
        "sections": sorted(k for k in sections if k != "_header"),
    }

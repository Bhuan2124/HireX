"""OCR for scanned resumes.

Many resumes arrive as a photo or scan saved inside a PDF: the page is one big
image and there is no text layer at all. PyMuPDF then returns an empty string,
and the resume scores as if it said nothing (0 skills, 0 years, no degree).

Two OCR engines are supported:

* RapidOCR (app/services/ocr_rapid.py) - installed by requirements.txt with
  its models inside the package. Nothing else to download or install.
* Tesseract - an optional separate program (on Windows:
  https://github.com/UB-Mannheim/tesseract/wiki). Slightly more accurate on the
  scanned Java resumes, so it is used automatically when it is installed.

RESUMEX_OCR_ENGINE = auto (default: Tesseract if installed, else RapidOCR) | rapidocr | tesseract
"""
from __future__ import annotations

import io
import os
import re
import shutil
import threading

from app.config import OCR_ENABLED, OCR_ENGINE, OCR_LANG, OCR_TARGET_WIDTH, TESSERACT_CMD
from app.services import ocr_rapid

# Several resumes are OCR'd in parallel; stop each Tesseract process from also
# spawning one thread per core, which would only make them fight for the CPU.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# Where the usual installers put tesseract.exe (UB-Mannheim installer, winget,
# Chocolatey, Scoop, per-user installs). Checked after TESSERACT_CMD and PATH.
_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\scoop\apps\tesseract\current\tesseract.exe"),
    os.path.expandvars(r"%ProgramData%\chocolatey\bin\tesseract.exe"),
    "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract",
]

INSTALL_HINT = ("Run 'pip install -r requirements.txt' (it installs the RapidOCR engine, "
                "nothing else is needed), then screen again. Check with: python tools/check_ocr.py")
TESSERACT_HINT = ("Install Tesseract OCR (Windows: https://github.com/UB-Mannheim/tesseract/wiki, "
                  "keep the default folder) and 'pip install pytesseract', or use the built-in "
                  "engine: RESUMEX_OCR_ENGINE=rapidocr.")


class OCRUnavailable(RuntimeError):
    """Raised when a page needs OCR but Tesseract cannot be used."""


_ready = None                   # pytesseract once it has worked; failures are retried
_lock = threading.Lock()


def active_engine():
    """-> (name or None, module/engine, reason it is unavailable).

    auto: Tesseract when it is installed (a little more accurate), otherwise
    RapidOCR, which needs nothing beyond requirements.txt.
    A working engine is remembered; a missing one is looked for again on the
    next call, so installing it works without restarting the server."""
    if not OCR_ENABLED:
        return None, None, "OCR is switched off (RESUMEX_OCR=0)."
    order = {"rapidocr": ["rapidocr"], "tesseract": ["tesseract"]}.get(OCR_ENGINE, ["tesseract", "rapidocr"])
    reasons = []
    for name in order:
        if name == "rapidocr":
            eng, why = ocr_rapid.load()
        else:
            eng, why = _engine()
        if eng is not None:
            return name, eng, ""
        reasons.append(why)
    hint = TESSERACT_HINT if order == ["tesseract"] else INSTALL_HINT
    return None, None, "No OCR engine available: " + " ".join(reasons) + " " + hint


def _engine():
    """-> (pytesseract module or None, reason it is unavailable).

    A working engine is remembered. A missing one is looked for again on the
    next call, so installing Tesseract works without restarting the server."""
    global _ready
    if _ready is not None:
        return _ready, ""
    with _lock:
        if _ready is not None:
            return _ready, ""
        tess, why = _find_engine()
        if tess is not None:
            _ready = tess
        return tess, why


def candidate_paths():
    """Every place tesseract is looked for, in order (for tools/check_ocr.py)."""
    out = []
    if TESSERACT_CMD:
        out.append(("TESSERACT_CMD setting", TESSERACT_CMD))
    out.append(("PATH", shutil.which("tesseract") or "(not on PATH)"))
    out += [("default install folder", p) for p in _WINDOWS_PATHS]
    return out


def _find_engine():
    if not OCR_ENABLED:
        return None, "OCR is switched off (RESUMEX_OCR=0)."
    try:
        import pytesseract
        import PIL  # noqa: F401
    except ImportError:
        return None, "pytesseract is not installed."
    cmd = (TESSERACT_CMD or shutil.which("tesseract")
           or next((p for p in _WINDOWS_PATHS if os.path.isfile(p)), None))
    if not cmd:
        return None, "Tesseract is not installed."
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:                                    # noqa: BLE001
        return None, f"Tesseract at {cmd} could not run ({exc})."
    _info["version"] = str(version)
    _info["path"] = cmd
    return pytesseract, ""


_info: dict = {}


def status() -> dict:
    """For the upload page: is OCR ready, which engine, and if not, why."""
    name, _, why = active_engine()
    label = {"rapidocr": "RapidOCR " + ocr_rapid.version(),
             "tesseract": "Tesseract " + _info.get("version", "")}.get(name, "")
    return {"available": name is not None, "engine": name or "", "label": label.strip(),
            "reason": why, "version": _info.get("version", ""), "lang": OCR_LANG}


def available() -> bool:
    return active_engine()[0] is not None


def _flatten(img):
    """Upright RGB image on a white background (no enlargement: RapidOCR
    reads small scans better at their own size)."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, "white")
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    return img.convert("RGB")


def _prepare(img):
    """Greyscale, upscale small scans with a smooth filter, then sharpen.

    Most scanned resumes are small JPEGs (500-1000 px wide). Tesseract wants
    text roughly 30 px tall, so a 544 px page is enlarged ~4.5x. LANCZOS keeps
    the letter edges smooth; nearest-neighbour or plain bilinear enlargement
    leaves blocky JPEG artefacts that Tesseract misreads badly.
    """
    from PIL import Image, ImageFilter, ImageOps

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, "white")
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    img = img.convert("L")
    if img.width < OCR_TARGET_WIDTH:
        scale = OCR_TARGET_WIDTH / img.width
        img = img.resize((OCR_TARGET_WIDTH, round(img.height * scale)), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return img


def image_to_layout(img) -> tuple[str, list[tuple[str, float]]]:
    """OCR one PIL image -> (text, lines).

    lines is [(line_text, letter_height_px)] in reading order. The height is what
    lets the parser find the name (the biggest text on a resume) and the job
    title (usually the next biggest) instead of guessing from line order, which
    two-column layouts scramble. Raises OCRUnavailable if Tesseract is missing.
    """
    name, eng, why = active_engine()
    if name is None:
        raise OCRUnavailable(why)
    if name == "rapidocr":
        return ocr_rapid.image_to_layout(_flatten(img), _font_height)
    tess = eng
    prepared = _prepare(img)
    d = tess.image_to_data(prepared, lang=OCR_LANG, config="--oem 3 --psm 3",
                           output_type=tess.Output.DICT)
    if _ink_coverage(prepared, d) < 0.75:
        # Tesseract's automatic layout analysis sometimes skips whole blocks of
        # small text (e.g. the name/contact header of a tiny scan). When much of
        # the inked page has no words, try the single-column mode and keep it if
        # it reads clearly more.
        d4 = tess.image_to_data(prepared, lang=OCR_LANG, config="--oem 3 --psm 4",
                                output_type=tess.Output.DICT)
        if _confident_words(d4) > 1.1 * _confident_words(d):
            d = d4
    scale = OCR_TARGET_WIDTH / max(1, prepared.width)     # comparable across pages
    groups: dict[tuple, list[tuple[str, int]]] = {}
    order: list[tuple] = []
    for i, word in enumerate(d["text"]):
        word = (word or "").strip()
        if not word or str(d["conf"][i]) == "-1":
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((word, _font_height(word, int(d["height"][i]))))

    text_parts, lines, prev = [], [], None
    for key in order:
        words = groups[key]
        if prev is not None:
            text_parts.append("\n\n" if key[:2] != prev[:2] else "\n")
        text_parts.append(" ".join(w for w, _ in words))
        prev = key
        # Tesseract often puts text of different sizes on one line ("Robert
        # Smith" in 40pt next to "Java Developer" in 16pt next to a phone
        # number). Split such a line into runs of similar height, each with its
        # own size (median word height of the run).
        run = [words[0]]
        for w in words[1:]:
            ref = sorted(h for _, h in run)[len(run) // 2]
            if 0.65 * ref <= w[1] <= ref / 0.65:
                run.append(w)
            else:
                lines.append(_run(run, scale))
                run = [w]
        lines.append(_run(run, scale))
    return "".join(text_parts), lines


def _confident_words(d):
    return sum(1 for t, c in zip(d["text"], d["conf"]) if str(t).strip() and float(c) > 60)


def _ink_coverage(img, d):
    """Share of the page's inked rows that lie inside a recognised word."""
    import numpy as np

    ink = (np.asarray(img) < 110).sum(axis=1) > img.width * 0.004
    covered = np.zeros(img.height, bool)
    for t, top, h, c in zip(d["text"], d["top"], d["height"], d["conf"]):
        if str(t).strip() and float(c) >= 0:
            covered[top:top + h] = True
    k = max(1, img.height // 200)                      # allow for line spacing
    covered = np.convolve(covered, np.ones(2 * k + 1), "same") > 0
    return float((ink & covered).sum()) / max(1, int(ink.sum()))


_TALL = re.compile(r"[A-Z0-9bdfhiklt\[\](){}|/\\!?#$%&@]")
_DESC = re.compile(r"[gjpqyQ,;()\[\]{}|/]")


def _font_height(word, box_h):
    """Box height -> comparable text size. A word box spans from the tallest
    letter's top to the lowest letter's bottom, so "Capital" (capital + p
    descender) measures ~30% taller than "WILLIS" at the same font size."""
    top = 1.0 if _TALL.search(word) else 0.72
    bottom = 0.28 if _DESC.search(word) else 0.0
    return box_h / (top + bottom)


def _run(words, scale):
    hs = sorted(h for _, h in words)
    return " ".join(w for w, _ in words), hs[len(hs) // 2] * scale


def image_to_text(img) -> str:
    """OCR one PIL image. Raises OCRUnavailable if no OCR engine is available."""
    return image_to_layout(img)[0]


def image_file_to_layout(path: str):
    """OCR a .png / .jpg / .jpeg / .tif / .bmp / .webp resume (every frame of a TIFF).
    -> (text, lines of the first page)"""
    from PIL import Image, ImageSequence

    with Image.open(path) as im:
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
    texts, first_lines = [], []
    for i, f in enumerate(frames):
        t, lines = image_to_layout(f)
        texts.append(t)
        if i == 0:
            first_lines = lines
    return "\n\n".join(texts), first_lines


def image_file_to_text(path: str) -> str:
    return image_file_to_layout(path)[0]


def pdf_page_to_layout(page):
    """OCR one PyMuPDF page -> (text, lines). See pdf_page_to_text."""
    return image_to_layout(_render(page))


def pdf_page_to_text(page) -> str:
    return pdf_page_to_layout(page)[0]


def _render(page):
    """Render one PyMuPDF page to a PIL image for OCR.

    The page is rendered at the native resolution of its largest embedded image
    (the scan), so no detail is invented or lost at this step; _prepare() then
    does the enlargement with a proper filter.
    """
    from PIL import Image

    dpi = 200
    try:
        infos = page.get_image_info()
    except Exception:                                           # noqa: BLE001
        infos = []
    if infos:
        def area(i):
            x0, y0, x1, y1 = i["bbox"]
            return (x1 - x0) * (y1 - y0)
        big = max(infos, key=area)
        width_in = (big["bbox"][2] - big["bbox"][0]) / 72.0
        if width_in > 0 and big.get("width"):
            dpi = big["width"] / width_in
    dpi = int(max(72, min(dpi, 300)))
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

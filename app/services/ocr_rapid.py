"""Pure-pip OCR engine: RapidOCR (PaddleOCR models on onnxruntime).

    pip install rapidocr_onnxruntime wordninja

Nothing else to install or download: the detection and recognition models are
inside the rapidocr_onnxruntime wheel, and wordninja ships its word list.

Two things are added on top of the raw engine:

* Reading order. RapidOCR returns one box per text line fragment, in no useful
  order. The boxes are put in reading order with a recursive XY-cut: columns
  (a sidebar, a date column) are read one after the other instead of being
  interleaved line by line.
* Missing spaces. The bundled recognition model is trained on Chinese and
  English text and sometimes drops the spaces between English words
  ("SENIORCOREJAVADEVELOPER"). Long letter runs are split back into words with
  wordninja's English word frequencies, but only when every piece is a sensible
  word, so surnames ("Vandervort") and tech names ("Kubernetes") stay intact.
"""
from __future__ import annotations

import re
import threading

_engine = None
_engine_error = ""
_lock = threading.Lock()
_run_lock = threading.Lock()
_split = None                      # wordninja.split, if installed


def load():
    """-> (engine or None, reason). The engine is created once and shared."""
    global _engine, _engine_error, _split
    if _engine is not None:
        return _engine, ""
    with _lock:
        if _engine is not None:
            return _engine, ""
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return None, "rapidocr_onnxruntime is not installed (pip install rapidocr_onnxruntime)."
        try:
            _engine = RapidOCR()
        except Exception as exc:                               # noqa: BLE001
            _engine_error = f"RapidOCR could not start ({exc})."
            return None, _engine_error
        try:
            import wordninja
            _split = wordninja.split
        except ImportError:
            _split = None
        return _engine, ""


def version() -> str:
    try:
        from importlib.metadata import version as v
        return v("rapidocr_onnxruntime")
    except Exception:                                          # noqa: BLE001
        return ""


# ------------------------------------------------------------------ OCR
def image_to_layout(img, font_height) -> tuple[str, list[tuple[str, float]]]:
    """PIL image -> (text, [(line text, text size)]) like the Tesseract engine."""
    import numpy as np

    engine, why = load()
    if engine is None:
        raise RuntimeError(why)
    rgb = np.asarray(img.convert("RGB"))
    with _run_lock:          # one page at a time; onnxruntime already uses every core
        result, _ = engine(rgb[:, :, ::-1].copy())      # RapidOCR expects BGR
    if not result:
        return "", []
    scale = 2500.0 / max(1, img.width)                  # sizes comparable across pages
    items = []
    for box, text, score in result:
        text = fix_spacing(str(text).strip())
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append((min(xs), min(ys), max(xs), max(ys), text))

    ordered_blocks = _xy_cut(items, img.width)
    parts, lines = [], []
    for block in ordered_blocks:
        rows = _rows(block)
        parts.append("\n".join("  ".join(t for *_, t in row) for row in rows))
        for row in rows:
            for x0, y0, x1, y1, t in row:
                lines.append((t, font_height(t, (y1 - y0) * 0.75) * scale))
    return "\n\n".join(parts), lines


# ------------------------------------------------------------------ reading order
def _xy_cut(items, page_w):
    """Reading order -> list of blocks: [header rows, left column, right column, footer rows].

    Resumes have at most one real column break: a sidebar running down most of
    the page, often under a full-width header and above a full-width footer.
    That gutter is found; everything else is read row by row, so a right-aligned
    date stays on the same line as its job title.
    """
    if len(items) < 6:
        return [items]
    best = None
    for k in range(15, 86):
        x = page_w * k / 100.0
        crossing = sorted((i for i in items if i[0] < x < i[2]), key=lambda i: i[1])
        # the tallest stretch of the page that no box crosses at x
        edges = [(-1e9, -1e9)] + [(i[1], i[3]) for i in crossing] + [(1e9, 1e9)]
        lo, hi, span = None, None, -1
        for (a0, a1), (b0, b1) in zip(edges, edges[1:]):
            if b0 - a1 > span:
                lo, hi, span = a1, b0, b0 - a1
        mid = [i for i in items if i[1] >= lo and i[3] <= hi]
        left = [i for i in mid if i[2] <= x]
        right = [i for i in mid if i[0] >= x]
        if len(left) < 3 or len(right) < 3 or len(mid) < 0.6 * len(items):
            continue
        lw = max(i[2] for i in left) - min(i[0] for i in left)
        rw = max(i[2] for i in right) - min(i[0] for i in right)
        gutter = min(i[0] for i in right) - max(i[2] for i in left)
        if lw < 0.2 * page_w or rw < 0.2 * page_w or gutter < 0.01 * page_w:
            continue
        # a sidebar sits beside the main text along its height, not as a few
        # short right-aligned entries (dates): both sides must run down the page
        lh = max(i[3] for i in left) - min(i[1] for i in left)
        rh = max(i[3] for i in right) - min(i[1] for i in right)
        if min(lh, rh) < 0.5 * (hi - lo if hi < 1e8 else max(lh, rh)):
            continue
        score = (len(mid), gutter)
        if best is None or score > best[0]:
            best = (score, lo, hi, x)
    if best is None:
        return [items]
    _, lo, hi, x = best
    mid = [i for i in items if i[1] >= lo and i[3] <= hi]
    head = [i for i in items if i[3] <= lo or (i not in mid and i[1] < lo)]
    foot = [i for i in items if i not in mid and i not in head]
    return [b for b in (head, [i for i in mid if i[2] <= x], [i for i in mid if i[2] > x], foot) if b]


def _rows(block):
    """Group a block's boxes into rows (vertical overlap), left to right."""
    rows = []
    for it in sorted(block, key=lambda i: (i[1], i[0])):
        h = it[3] - it[1]
        for row in rows:
            r0, r1 = min(i[1] for i in row), max(i[3] for i in row)
            overlap = min(r1, it[3]) - max(r0, it[1])
            if overlap > 0.5 * min(h, r1 - r0):
                row.append(it)
                break
        else:
            rows.append([it])
    return [sorted(r, key=lambda i: i[0]) for r in rows]


# ------------------------------------------------------------------ spacing repair
_SHORT_OK = set("a i an as at be by do go he if in is it me my no of on or so to up us we "
                "ai ui qa hr js db ml os pc ux id vs ok tv".split())
_RUN = re.compile(r"(?<=\')s[A-Za-z]{9,}|[A-Za-z]{10,}")
_protected: set[str] | None = None


def _protected_words():
    """Skill names written as one word (postgresql, kubernetes, javascript...)."""
    global _protected
    if _protected is None:
        from app.services.ontology import ONTOLOGY
        out = set()
        for canon, (_, variants) in ONTOLOGY.items():
            for s in [canon] + list(variants):
                out.add(re.sub(r"[^a-z]", "", s.lower()))
        _protected = out
    return _protected


def _split_run(run: str) -> str:
    if _split is None or run.lower() in _protected_words():
        return run
    pieces = _split(run)
    if len(pieces) < 2:
        return run
    lead = ""
    if pieces[0] == "s" and len(pieces) > 1:      # "Bachelor's" + "sDegreein..." keeps its s
        lead, pieces = "s", pieces[1:]
    merged = []                                   # "deliverable" + "s" -> "deliverables"
    for p in pieces:
        if merged and p.lower() == "s":
            merged[-1] += p
        else:
            merged.append(p)
    for p in merged:
        if len(p) <= 2 and p.lower() not in _SHORT_OK:
            return run                            # "Vande rv ort": not real words
    if len(lead) + sum(len(p) for p in merged) != len(run):
        return run
    return (lead + " " if lead else "") + " ".join(merged)


def fix_spacing(text: str) -> str:
    """Re-insert spaces the recogniser dropped between English words."""
    if _split is None:
        return text
    return _RUN.sub(lambda m: _split_run(m.group(0)), text)

"""Check that OCR (needed for scanned resumes) works on this computer.

    python tools/check_ocr.py

Shows which OCR engine ResumeX will use, reads a small test image with it,
and says what to do if something is missing.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[X] Pillow is missing. Fix: pip install -r requirements.txt")
        return 1

    from app.services import ocr, ocr_rapid

    eng, why = ocr_rapid.load()
    print(f"[{'OK' if eng else ' X'}] RapidOCR (built in){': ' + ocr_rapid.version() if eng else ' - ' + why}")
    tess, twhy = ocr._engine()
    print(f"[{'OK' if tess else '--'}] Tesseract (optional){': ' + ocr._info.get('version', '') if tess else ' - ' + twhy}")

    st = ocr.status()
    if not st["available"]:
        print(f"\n[X] No OCR engine available.\n    Fix: pip install -r requirements.txt  (Python 3.11 or 3.12)")
        if sys.version_info >= (3, 13):
            print("    Your Python is 3.13 or newer; the built-in engine needs 3.12 or older.")
        return 1
    print(f"\nResumeX will use: {st['label']}"
          f"{'  (set RESUMEX_OCR_ENGINE=rapidocr or tesseract to choose)' if os.getenv('RESUMEX_OCR_ENGINE') is None else ''}")

    img = Image.new("RGB", (1000, 140), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 48)
            break
        except OSError:
            continue
    draw.text((20, 40), "Senior Java Developer 2018 - 2022", fill="black", font=font or ImageFont.load_default())
    text = ocr.image_to_text(img).strip()
    print(f"[OK] Test image read as: {text!r}")
    if "java" not in text.lower():
        print("[!] The test text was not read correctly.")
        return 1
    print("\nOCR works. Restart ResumeX if it is running, then upload the resumes again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ResumeX container image.
#
# Sentence-BERT is downloaded during the build, so the running container starts
# in seconds and never needs the internet. Torch is the CPU-only build, which is
# about a tenth the size of the default GPU one.
#
# Railway, Render, Cloud Run and Hugging Face Spaces build this file directly.
#
# Build and run locally:
#   docker build -t resumex .
#   docker run -p 8080:8080 -e PORT=8080 resumex
# Then open http://localhost:8080

# Pinned to Debian bookworm so the system package names below stay valid.
FROM python:3.11-slim-bookworm

# Keep the image lean and predictable at runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# - build-essential: build basics for a few wheels (removed again below)
# - tesseract-ocr: OCR for scanned resumes. In the container it is preferred
#   over the pip-only RapidOCR engine: a little more accurate and lighter on
#   RAM, which matters on small plans (Railway Trial = 1 GB).
# - libgl1, libglib2.0-0: needed by OpenCV, which RapidOCR (the fallback) uses
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. CPU-only PyTorch first, from PyTorch's own index (~200 MB instead of ~2 GB).
#    Unpinned on purpose: this index only carries recent builds, and a pin that
#    ages out breaks the build months later.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else. Copied separately so edits to the app don't reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && python -c "import cv2, rapidocr_onnxruntime, pytesseract, fitz; print('OCR stack OK, tesseract', pytesseract.get_tesseract_version())"

# 3. Bake Sentence-BERT into the image. This is the step that makes the deployed
#    app start fast: no 90 MB download on the first request.
ENV SBERT_LOCAL_DIR=/app/ml/sbert_model
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
m.save('/app/ml/sbert_model'); \
print('SBERT baked into the image')"

# 4. The application itself.
COPY . .

# Database and uploads live in RESUMEX_STATE_DIR. /tmp is wiped on every
# redeploy; on Railway attach a volume at /data and set RESUMEX_STATE_DIR=/data
# (see deploy/RAILWAY.md) so jobs and candidates survive.
# PORT is set here so the app knows it is deployed even on platforms that don't
# set it themselves (Hugging Face Spaces). Railway and Cloud Run override it.
# Two resumes are OCR'd at a time: enough for 1-2 vCPU, safe for 1 GB of RAM.
ENV RESUMEX_STATE_DIR=/tmp/resumex \
    RESUMEX_PARSE_WORKERS=2 \
    PORT=8080
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

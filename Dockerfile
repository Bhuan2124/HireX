# ResumeX container image.
#
# Sentence-BERT is downloaded during the build, so the running container starts
# in seconds and never needs the internet. Torch is the CPU-only build, which is
# about a tenth the size of the default GPU one.
#
# Build and run locally:
#   docker build -t resumex .
#   docker run -p 8080:8080 -e PORT=8080 resumex
# Then open http://localhost:8080

FROM python:3.11-slim

# Keep the image lean and predictable at runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# PyMuPDF and friends need nothing exotic, but pip needs build basics for a few
# wheels. Removed again in the same layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. CPU-only PyTorch first, from PyTorch's own index (~200 MB instead of ~2 GB).
#    Unpinned on purpose: this index only carries recent builds, and a pin that
#    ages out breaks the build months later.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else. Copied separately so edits to the app don't reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

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

# Database and uploads live here; the project folder stays read-only.
# PORT is set here so the app knows it is deployed even on platforms that don't
# set it themselves (Hugging Face Spaces). Cloud Run overrides it with its own.
ENV RESUMEX_STATE_DIR=/tmp/resumex \
    PORT=8080
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

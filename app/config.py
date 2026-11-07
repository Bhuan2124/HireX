"""Central configuration. Edit here, or override with environment variables.

Running locally, the defaults below are all you need. Deployed, the platform
sets PORT and you set HIREX_SECRET (and DATABASE_URL if you add Postgres);
everything else adapts on its own.
"""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# True when running on a hosting platform rather than your laptop. Cloud Run,
# Hugging Face Spaces and most others set PORT; nothing sets it locally.
IS_DEPLOYED = bool(os.getenv("PORT"))

# Writable location for the database and uploads. A deployed container often has
# a read-only project folder, so those go to /tmp there.
STATE_DIR = Path(os.getenv("HIREX_STATE_DIR", "/tmp/hirex" if IS_DEPLOYED else BASE_DIR))

# ---------------------------------------------------------------- database
# Default: SQLite, created automatically, no server to install.
# Deployed on a container platform, SQLite is wiped whenever the container
# restarts. For data that survives, create a free Postgres (Neon or Supabase)
# and set DATABASE_URL to its connection string - nothing else changes.
# To use MySQL instead: pip install pymysql, create the schema, then set
#   DATABASE_URL = "mysql+pymysql://root:yourpassword@localhost:3306/hirex"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STATE_DIR / 'hirex.db'}")
# Railway / Heroku-style Postgres URLs start with postgres://, which SQLAlchemy
# no longer accepts; the driver is psycopg2 (in requirements.txt).
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len("postgres://"):]
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len("postgresql://"):]

# ---------------------------------------------------------------- paths
UPLOAD_DIR = Path(os.getenv("HIREX_UPLOAD_DIR", STATE_DIR / "uploads"))
DATA_DIR = BASE_DIR / "data"            # HireX dataset CSVs go here
MODEL_DIR = BASE_DIR / "ml" / "models"
STATE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
for _d in (DATA_DIR, MODEL_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:                      # read-only image layer: already shipped
        pass

# ---------------------------------------------------------------- app
# Set HIREX_SECRET when you deploy. Without it a random key is generated at
# startup, which works but logs everyone out whenever the server restarts.
SECRET_KEY = os.getenv("HIREX_SECRET") or ("change-this-secret-key"
                                             if not IS_DEPLOYED else secrets.token_hex(32))
MAX_UPLOAD_MB = int(os.getenv("HIREX_MAX_UPLOAD_MB", "10"))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"} | IMAGE_EXTENSIONS

# ---------------------------------------------------------------- OCR (scanned resumes)
# Scanned PDFs and photos have no text layer; they are read with Tesseract OCR.
# Tesseract is found on PATH or in its default Windows folder; otherwise set
# TESSERACT_CMD to the full path of tesseract(.exe). HIREX_OCR=0 switches OCR off.
OCR_ENABLED = os.getenv("HIREX_OCR", "1") != "0"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
OCR_LANG = os.getenv("HIREX_OCR_LANG", "eng")
OCR_ENGINE = os.getenv("HIREX_OCR_ENGINE", "auto").lower()   # auto | rapidocr | tesseract
OCR_TARGET_WIDTH = 2500      # small scans are enlarged to about this width before OCR
OCR_MIN_PAGE_CHARS = 40      # a PDF page with less text than this (and an image) is OCR'd
# Resumes parsed at the same time during screening (OCR is CPU-bound, ~2-4 s a page).
PARSE_WORKERS = int(os.getenv("HIREX_PARSE_WORKERS", str(max(1, min(4, os.cpu_count() or 2)))))

# ---------------------------------------------------------------- semantic matching
# HireX runs on Sentence-BERT. There is no fallback: if the model cannot load,
# the server refuses to start and prints exactly what to fix.
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Saved here by `python tools/download_model.py`, so the app runs offline afterwards.
# The Docker build bakes the model into the image at this same path.
SBERT_LOCAL_DIR = Path(os.getenv("SBERT_LOCAL_DIR", BASE_DIR / "ml" / "sbert_model"))
SBERT_DEVICE = os.getenv("SBERT_DEVICE", "auto")      # auto | cpu | cuda
SBERT_BATCH_SIZE = 32

# MiniLM reads at most 256 word-pieces, roughly 180 words. A whole resume is far
# longer, so it is split into overlapping chunks and each chunk is embedded.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30
TOP_K_CHUNKS = 3        # resume similarity = mean of its 3 best-matching chunks

# Raw SBERT cosine for resume-vs-job pairs sits roughly between these values.
# They are mapped linearly onto 0-1 for the score. The upload screen shows the
# raw cosines, so check them on your own data and adjust if needed.
SIM_FLOOR = 0.15
SIM_CEIL = 0.65

# A missing skill counts as having a "related skill" when the candidate has a
# skill whose embedding is at least this similar to it.
RELATED_SKILL_MIN = 0.55

# Default recruiter weights (must sum to 1.0); editable per job in the UI.
# "practical" = internships + projects. A fresher has no full-time experience,
# so for fresher jobs that weight moves to practical work instead.
DEFAULT_WEIGHTS = {"skills": 0.40, "experience": 0.25, "practical": 0.0,
                   "education": 0.15, "semantic": 0.20}
WEIGHTS_BY_LEVEL = {
    "Experienced": {"skills": 0.40, "experience": 0.25, "practical": 0.00, "education": 0.15, "semantic": 0.20},
    "Fresher":     {"skills": 0.35, "experience": 0.00, "practical": 0.30, "education": 0.15, "semantic": 0.20},
    "Any":         {"skills": 0.35, "experience": 0.15, "practical": 0.15, "education": 0.15, "semantic": 0.20},
}
EXPERIENCE_LEVELS = ["Fresher", "Experienced", "Any"]

# ---------------------------------------------------------------- practical work (freshers)
INTERNSHIP_FULL_MONTHS = 6       # an internship this long (or longer) gets full duration marks
INTERNSHIP_UNKNOWN_MONTHS = 2    # internship found but no dates: counted as this many months
# best project counts 60%, 2nd best 25%, 3rd best 15% (more good projects = higher score)
PROJECT_RANK_WEIGHTS = [0.60, 0.25, 0.15]
# only one of the two present (no internship, or no projects): it counts for this much
PRACTICAL_SINGLE_SHARE = 0.80

# ---------------------------------------------------------------- two-sided platform
# How a candidate's match is labelled before applying and on their application page.
FIT_LABELS = [(0.70, "Strong match"), (0.50, "Good match"), (0.30, "Low match"), (0.0, "Weak match")]
# A resume with no must-have skill AND SBERT fit below this is "Not eligible" (different field).
NOT_ELIGIBLE_SEMANTIC = 0.25
# Interview zone shown to candidates after the deadline = openings x this (unless the recruiter sets it).
SHORTLIST_MULTIPLIER = 3
REJECT_REASONS = ["Skills gap", "Experience below requirement", "Education below requirement",
                  "Not a fit for this role", "Positions filled", "After interview", "Other"]
# Two applications to the same job sharing this much of their wording are flagged as copies.
DUPLICATE_THRESHOLD = 0.60

EDUCATION_LEVELS = {"Any": 0, "High School": 1, "Associate/Diploma": 2,
                    "Bachelor": 3, "Master": 4, "Doctorate": 5}

CATEGORIES = ["INFORMATION-TECHNOLOGY", "ENGINEERING", "FINANCE", "ACCOUNTANT", "BANKING",
              "HEALTHCARE", "HR", "SALES", "BUSINESS-DEVELOPMENT", "CONSULTANT", "ADVOCATE",
              "TEACHER", "DESIGNER", "DIGITAL-MEDIA", "PUBLIC-RELATIONS", "CONSTRUCTION",
              "AUTOMOBILE", "AVIATION", "AGRICULTURE", "CHEF", "FITNESS", "APPAREL", "ARTS", "BPO"]

STAGES = ["Applied", "AI Screened", "Shortlisted", "Interview", "Selected", "Rejected"]

REQ_CATEGORIES = ["Must Have", "Preferred", "Nice to Have"]
REQ_WEIGHT = {"Must Have": 1.0, "Preferred": 0.6, "Nice to Have": 0.3}

# A candidate evidencing none of the must-have skills has their weighted score
# multiplied by this factor. Shown in the explanation, never applied silently.
NO_MUST_HAVE_PENALTY = 0.5

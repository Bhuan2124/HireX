"""Central configuration. Edit here, or override with environment variables.

Running locally, the defaults below are all you need. Deployed, the platform
sets PORT and you set RESUMEX_SECRET (and DATABASE_URL if you add Postgres);
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
STATE_DIR = Path(os.getenv("RESUMEX_STATE_DIR", "/tmp/resumex" if IS_DEPLOYED else BASE_DIR))

# ---------------------------------------------------------------- database
# Default: SQLite, created automatically, no server to install.
# Deployed on a container platform, SQLite is wiped whenever the container
# restarts. For data that survives, create a free Postgres (Neon or Supabase)
# and set DATABASE_URL to its connection string - nothing else changes.
# To use MySQL instead: pip install pymysql, create the schema, then set
#   DATABASE_URL = "mysql+pymysql://root:yourpassword@localhost:3306/resumex"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STATE_DIR / 'resumex.db'}")

# ---------------------------------------------------------------- paths
UPLOAD_DIR = Path(os.getenv("RESUMEX_UPLOAD_DIR", STATE_DIR / "uploads"))
DATA_DIR = BASE_DIR / "data"            # ResumeX dataset CSVs go here
MODEL_DIR = BASE_DIR / "ml" / "models"
STATE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
for _d in (DATA_DIR, MODEL_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:                      # read-only image layer: already shipped
        pass

# ---------------------------------------------------------------- app
# Set RESUMEX_SECRET when you deploy. Without it a random key is generated at
# startup, which works but logs everyone out whenever the server restarts.
SECRET_KEY = os.getenv("RESUMEX_SECRET") or ("change-this-secret-key"
                                             if not IS_DEPLOYED else secrets.token_hex(32))
MAX_UPLOAD_MB = int(os.getenv("RESUMEX_MAX_UPLOAD_MB", "10"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ---------------------------------------------------------------- semantic matching
# ResumeX runs on Sentence-BERT. There is no fallback: if the model cannot load,
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
DEFAULT_WEIGHTS = {"skills": 0.40, "experience": 0.25, "education": 0.15, "semantic": 0.20}

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

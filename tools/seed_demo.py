"""Fill HireX with demo data so both sides can be shown in a review.

    python tools/seed_demo.py            # adds the demo accounts, jobs and applications
    python tools/seed_demo.py --reset    # deletes hirex.db first (start clean)

Everyone's password is demo123. Recruiter: hr@acme.com.
Candidates: ananya@, sneha@, rahul@, vikram@, priya@, anu@demo.com
What each one shows is described in app/services/demo.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if "--reset" in sys.argv:
    from app.config import DATABASE_URL
    if DATABASE_URL.startswith("sqlite:///"):
        Path(DATABASE_URL.replace("sqlite:///", "")).unlink(missing_ok=True)
        print("Deleted the old database.")

from app.services.demo import seed  # noqa: E402

if __name__ == "__main__":
    seed()
    print("Start the app with: python run.py")

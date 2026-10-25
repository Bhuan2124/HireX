"""Export sample resumes from the dataset as .txt files so you can test uploading.

    python tools/make_test_resumes.py            # 25 mixed resumes
    python tools/make_test_resumes.py 40 INFORMATION-TECHNOLOGY

Files land in test_resumes/. Upload them through the UI exactly like real files.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "resumes.csv"
OUT = ROOT / "test_resumes"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    category = sys.argv[2].upper() if len(sys.argv) > 2 else None
    if not SRC.exists():
        sys.exit(f"Missing {SRC} - copy the dataset CSVs into data/ first.")

    df = pd.read_csv(SRC, usecols=["Resume_ID", "Category", "Current_Job_Title", "Resume_Text"]).fillna("")
    if category:
        df = df[df.Category == category]
        if df.empty:
            sys.exit(f"No resumes in category {category}")
    df = df.sample(n=min(n, len(df)), random_state=42)

    OUT.mkdir(exist_ok=True)
    for r in df.itertuples():
        title = (str(r.Current_Job_Title) or "candidate").replace("/", "-")[:40].strip()
        name = f"{r.Resume_ID}_{title or 'candidate'}.txt".replace(" ", "_")
        (OUT / name).write_text(str(r.Resume_Text), encoding="utf-8")
    print(f"Wrote {len(df)} resumes to {OUT}")
    print("Now upload them on the ResumeX upload screen.")


if __name__ == "__main__":
    main()

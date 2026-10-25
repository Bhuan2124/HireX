"""Train the classical models on the ResumeX dataset.

Run from the project root:      python ml/train_models.py

Inputs  (put these in data/):   resume_job_matches.csv, resumes.csv
Outputs (written to ml/models/): relevance_model.joblib, category_model.joblib,
                                 training_report.txt

Two models:
  1. Relevance  - component scores -> relevance label 1-5.
                  SVM and Random Forest are both trained and compared; the better
                  macro-F1 on validation is saved for the app to use.
  2. Category   - resume text -> job domain (24 classes). The app uses it to tell
                  whether a new resume is from the same field as the job.

Splits come from the dataset's own Split column, which is resume-level, so no
resume appears in both training and test.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "ml" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES = ["skill", "experience", "education", "project", "semantic", "domain"]
COLMAP = {"skill": "Skill_Match", "experience": "Experience_Match",
          "education": "Education_Match", "project": "Project_Match",
          "semantic": "Text_Similarity", "domain": "Domain_Match"}


def load_pairs() -> pd.DataFrame:
    path = DATA / "resume_job_matches.csv"
    if not path.exists():
        sys.exit(f"Missing {path}\nCopy the ResumeX dataset CSVs into the data/ folder first.")
    df = pd.read_csv(path)
    for short, col in COLMAP.items():
        df[short] = df[col]
    return df


def train_relevance(df: pd.DataFrame, report: list) -> dict:
    train = df[df.Split == "train"]
    val = df[df.Split == "val"]
    Xtr, ytr = train[FEATURES].to_numpy(), train.Relevance_Label.to_numpy()
    Xva, yva = val[FEATURES].to_numpy(), val.Relevance_Label.to_numpy()

    candidates = {
        "SVM (RBF)": Pipeline([("scale", StandardScaler()),
                               ("clf", SVC(C=4.0, gamma="scale", probability=True,
                                           class_weight="balanced", random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        pred = model.predict(Xva)
        f1 = f1_score(yva, pred, average="macro")
        results[name] = (model, f1)
        line = f"{name:15} val macro-F1 = {f1:.4f}   (fit {fit_s:.1f}s)"
        print("  " + line)
        report.append(line)

    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"  -> saving {best_name}")
    report.append(f"selected: {best_name}")
    report.append("\nvalidation report for the selected model:")
    report.append(classification_report(yva, best_model.predict(Xva), digits=3))

    if hasattr(best_model, "feature_importances_"):
        imp = dict(zip(FEATURES, np.round(best_model.feature_importances_, 4)))
        report.append(f"feature importances: {imp}")

    joblib.dump({"model": best_model, "name": best_name, "features": FEATURES},
                MODELS / "relevance_model.joblib")
    # keep both for the comparison table in the report
    for name, (model, f1) in results.items():
        slug = name.split()[0].lower()
        joblib.dump({"model": model, "name": name, "features": FEATURES},
                    MODELS / f"relevance_{slug}.joblib")
    return {"best": best_name, "scores": {k: v[1] for k, v in results.items()}}


def train_category(report: list):
    path = DATA / "resumes.csv"
    if not path.exists():
        print("  resumes.csv not found - skipping the category model")
        return
    res = pd.read_csv(path, usecols=["Resume_Text", "Category", "Split"]).fillna("")
    tr = res[res.Split == "train"]
    va = res[res.Split == "val"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=3, max_features=60000, sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, class_weight="balanced")),
    ])
    t0 = time.time()
    pipe.fit(tr.Resume_Text, tr.Category)
    acc = pipe.score(va.Resume_Text, va.Category)
    f1 = f1_score(va.Category, pipe.predict(va.Resume_Text), average="macro")
    line = f"Category model  val accuracy = {acc:.4f}  macro-F1 = {f1:.4f}  (fit {time.time()-t0:.1f}s)"
    print("  " + line)
    report.append(line)
    joblib.dump({"model": pipe, "name": "TF-IDF + LinearSVC"}, MODELS / "category_model.joblib")


def main():
    report = ["ResumeX model training report", "=" * 60, ""]
    print("\n[1/2] Relevance model (component scores -> label 1-5)")
    report.append("[1] Relevance model")
    df = load_pairs()
    print(f"  pairs: {len(df):,}  train/val/test = "
          f"{(df.Split=='train').sum():,}/{(df.Split=='val').sum():,}/{(df.Split=='test').sum():,}")
    report.append(f"pairs: {len(df):,}")
    train_relevance(df, report)

    print("\n[2/2] Category model (resume text -> domain)")
    report.append("\n[2] Category model")
    train_category(report)

    (MODELS / "training_report.txt").write_text("\n".join(str(r) for r in report))
    print(f"\nSaved to {MODELS}")
    print("Next:  python ml/evaluate.py")


if __name__ == "__main__":
    main()

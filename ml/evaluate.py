"""Evaluate every matching method on the held-out test split.

Run from the project root:   python ml/evaluate.py

Produces the results table for the report chapter:
  Accuracy, Precision, Recall, F1 (macro), NDCG@10, MRR, screening time.

Methods compared:
  Rule-based composite   the weighted score the product actually uses
  TF-IDF only            keyword baseline (Text_Similarity alone)
  Skill coverage only    single-component baseline
  SVM (RBF)              trained on component features
  Random Forest          trained on component features

The test split is resume-level, so no resume in it was seen during training.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "ml" / "models"
FEATURES = ["skill", "experience", "education", "project", "semantic", "domain"]
COLMAP = {"skill": "Skill_Match", "experience": "Experience_Match",
          "education": "Education_Match", "project": "Project_Match",
          "semantic": "Text_Similarity", "domain": "Domain_Match"}

# weights of the product's rule-based score (app/config.py DEFAULT_WEIGHTS)
RULE_W = {"skill": 0.40, "experience": 0.25, "education": 0.15, "semantic": 0.20}
GOOD = 4                      # labels >= 4 count as a relevant candidate


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    gains = relevance[order][:k]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(((2 ** gains - 1) * discounts).sum())
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevance: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    for pos, rel in enumerate(relevance[order], 1):
        if rel >= GOOD:
            return 1.0 / pos
    return 0.0


def ranking_metrics(df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    ndcgs, mrrs = [], []
    for _, g in df.groupby("Job_ID"):
        if len(g) < 2:
            continue
        rel = g.Relevance_Label.to_numpy().astype(float)
        sc = g[score_col].to_numpy().astype(float)
        ndcgs.append(ndcg_at_k(rel, sc, 10))
        mrrs.append(mrr(rel, sc))
    return float(np.mean(ndcgs)), float(np.mean(mrrs))


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def binary_from_score(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def main():
    path = DATA / "resume_job_matches.csv"
    if not path.exists():
        sys.exit(f"Missing {path} - copy the dataset CSVs into data/ first.")
    df = pd.read_csv(path)
    for short, col in COLMAP.items():
        df[short] = df[col]

    test = df[df.Split == "test"].copy()
    train = df[df.Split == "train"]
    print(f"test pairs: {len(test):,} across {test.Job_ID.nunique()} jobs\n")

    y_true_multi = test.Relevance_Label.to_numpy()
    y_true_bin = (y_true_multi >= GOOD).astype(int)
    rows = []

    # ---------------- unsupervised / rule-based scorers ----------------
    test["rule"] = sum(RULE_W[k] * test[k] for k in RULE_W)
    scorers = {"Rule-based composite": "rule",
               "TF-IDF similarity only": "semantic",
               "Skill coverage only": "skill"}
    for name, col in scorers.items():
        t0 = time.perf_counter()
        scores = test[col].to_numpy()
        elapsed = (time.perf_counter() - t0) * 1000
        # threshold chosen on TRAIN so the test split stays untouched
        tr_scores = (sum(RULE_W[k] * train[k] for k in RULE_W) if col == "rule"
                     else train[col]).to_numpy()
        tr_bin = (train.Relevance_Label.to_numpy() >= GOOD).astype(int)
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(tr_scores.min(), tr_scores.max(), 60):
            f1 = f1_score(tr_bin, (tr_scores >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        m = classification_metrics(y_true_bin, binary_from_score(scores, best_t))
        nd, mr = ranking_metrics(test.assign(_s=scores), "_s")
        rows.append({"Method": name, **m, "NDCG@10": nd, "MRR": mr,
                     "Time (ms/1k pairs)": elapsed / max(1, len(test)) * 1000})

    # ---------------- trained models ----------------
    for slug, label in [("svm", "SVM (RBF)"), ("random", "Random Forest")]:
        f = MODELS / f"relevance_{slug}.joblib"
        if not f.exists():
            print(f"  {label}: not trained yet (run python ml/train_models.py)")
            continue
        bundle = joblib.load(f)
        model = bundle["model"]
        X = test[FEATURES].to_numpy()
        t0 = time.perf_counter()
        pred = model.predict(X)
        elapsed = (time.perf_counter() - t0) * 1000
        m = classification_metrics((y_true_multi >= GOOD).astype(int),
                                   (pred >= GOOD).astype(int))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = np.array(model.classes_)
            score = proba[:, classes >= GOOD].sum(axis=1)
        else:
            score = pred.astype(float)
        nd, mr = ranking_metrics(test.assign(_s=score), "_s")
        multi = classification_metrics(y_true_multi, pred)
        rows.append({"Method": label, **m, "NDCG@10": nd, "MRR": mr,
                     "Time (ms/1k pairs)": elapsed / max(1, len(test)) * 1000})
        print(f"  {label}: 5-class accuracy {multi['Accuracy']:.4f}, "
              f"macro-F1 {multi['F1']:.4f}")

    res = pd.DataFrame(rows).set_index("Method").round(4)
    print("\n" + "=" * 92)
    print("RESULTS ON THE HELD-OUT TEST SPLIT  (relevant = label >= 4)")
    print("=" * 92)
    print(res.to_string())
    out = MODELS / "evaluation_results.csv"
    res.to_csv(out)
    print(f"\nSaved to {out}")
    print("Paste this table straight into the report's results chapter.")


if __name__ == "__main__":
    main()

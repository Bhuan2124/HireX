"""Loads the models trained by ml/train_models.py.

Two models, both optional: if the .joblib files are missing the app still runs on
the rule-based score alone, and the UI simply does not show the ML column.

  relevance_model.joblib  component scores -> relevance label 1-5 (SVM or RF)
  category_model.joblib   resume text      -> job domain (TF-IDF + LinearSVC)
"""
from __future__ import annotations

import threading
from pathlib import Path

from app.config import MODEL_DIR

FEATURES = ["skill", "experience", "education", "project", "semantic", "domain"]

_lock = threading.Lock()
_rel = _cat = None
_loaded = False


class RelevanceModel:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.name = bundle.get("name", "model")
        self.features = bundle.get("features", FEATURES)

    def predict(self, comp: dict) -> tuple[int, float]:
        """-> (predicted label 1-5, probability that the pair is a good match)"""
        import numpy as np
        x = np.array([[float(comp.get(f, 0.0)) for f in self.features]])
        label = int(self.model.predict(x)[0])
        prob = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            classes = list(self.model.classes_)
            prob = float(sum(p for c, p in zip(classes, proba) if c >= 4))
        else:
            prob = min(1.0, max(0.0, (label - 1) / 4))
        return label, round(prob, 4)


class CategoryModel:
    def __init__(self, bundle):
        self.pipeline = bundle["model"]

    def predict(self, text: str) -> str:
        if not text:
            return ""
        try:
            return str(self.pipeline.predict([text])[0])
        except Exception:                                    # noqa: BLE001
            return ""


def _load():
    global _rel, _cat, _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        import joblib
        rel_path = Path(MODEL_DIR) / "relevance_model.joblib"
        cat_path = Path(MODEL_DIR) / "category_model.joblib"
        if rel_path.exists():
            try:
                _rel = RelevanceModel(joblib.load(rel_path))
                print(f"[ml] relevance model loaded: {_rel.name}")
            except Exception as exc:                          # noqa: BLE001
                print(f"[ml] could not load relevance model: {exc}")
        else:
            print("[ml] no relevance model found - run: python ml/train_models.py")
        if cat_path.exists():
            try:
                _cat = CategoryModel(joblib.load(cat_path))
                print("[ml] category model loaded")
            except Exception as exc:                          # noqa: BLE001
                print(f"[ml] could not load category model: {exc}")
        _loaded = True


def get_relevance_model():
    _load()
    return _rel


def get_category_model():
    _load()
    return _cat


def status() -> dict:
    _load()
    return {"relevance_model": _rel.name if _rel else None,
            "category_model": bool(_cat)}

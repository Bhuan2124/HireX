"""Semantic matching with Sentence-BERT (all-MiniLM-L6-v2).

ResumeX runs on SBERT only. The model is loaded once when the server starts;
if it cannot load, startup fails with instructions instead of the app quietly
producing worse rankings.

Why chunking: MiniLM reads at most 256 word-pieces (~180 words) and silently
drops the rest. A resume is usually 500-1,500 words, so embedding it whole would
only ever look at its first paragraph. Each resume is split into overlapping
chunks, every chunk is embedded, and the resume's similarity to the job is the
mean of its best-matching chunks.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from app.config import (CHUNK_OVERLAP, CHUNK_WORDS, RELATED_SKILL_MIN,
                        SBERT_BATCH_SIZE, SBERT_DEVICE, SBERT_LOCAL_DIR,
                        SBERT_MODEL_NAME, SIM_CEIL, SIM_FLOOR, TOP_K_CHUNKS)

_lock = threading.Lock()
_model = None
_info: dict = {"loaded": False}
_skill_cache: dict[str, np.ndarray] = {}


class SBERTNotAvailable(RuntimeError):
    """Raised when Sentence-BERT cannot be loaded. The message says how to fix it."""


# ------------------------------------------------------------------ loading
def _pick_device() -> str:
    if SBERT_DEVICE in ("cpu", "cuda"):
        return SBERT_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                        # noqa: BLE001
        return "cpu"


def init() -> dict:
    """Load the model. Called once at server startup. Safe to call again."""
    global _model, _info
    if _model is not None:
        return _info
    with _lock:
        if _model is not None:
            return _info
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SBERTNotAvailable(
                "sentence-transformers is not installed.\n"
                "  Fix:  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "        pip install sentence-transformers\n"
                "        python tools/download_model.py") from exc

        device = _pick_device()
        local = Path(SBERT_LOCAL_DIR)
        t0 = time.time()
        try:
            if (local / "modules.json").exists():
                _model = SentenceTransformer(str(local), device=device)
                source = "local copy"
            else:
                # first run with internet: download, then keep a local copy
                _model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
                source = "downloaded"
                try:
                    local.mkdir(parents=True, exist_ok=True)
                    _model.save(str(local))
                    source = "downloaded and saved locally"
                except OSError:
                    pass                 # read-only filesystem (deployed): fine
        except Exception as exc:                             # noqa: BLE001
            raise SBERTNotAvailable(
                f"Could not load the SBERT model ({exc}).\n"
                "  The model has to be downloaded once with internet:\n"
                "        python tools/download_model.py\n"
                f"  It is then stored in {local} and works offline.") from exc

        _info = {
            "loaded": True,
            "model": SBERT_MODEL_NAME.split("/")[-1],
            "device": device,
            "dimension": int(_model.get_sentence_embedding_dimension()),
            "max_seq_length": int(getattr(_model, "max_seq_length", 256)),
            "source": source,
            "load_seconds": round(time.time() - t0, 1),
        }
        print(f"[SBERT] {_info['model']} ready on {device} "
              f"({_info['dimension']}-d, {_info['load_seconds']}s, {source})")
        return _info


def status() -> dict:
    return dict(_info)


def _require():
    if _model is None:
        init()
    return _model


# ------------------------------------------------------------------ primitives
def chunk(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows that fit the model's input limit."""
    words = (text or "").split()
    if not words:
        return [""]
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def encode(texts: list[str]) -> np.ndarray:
    """Unit-length sentence embeddings, shape (n, dim)."""
    model = _require()
    vecs = model.encode(texts, batch_size=SBERT_BATCH_SIZE, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def job_vector(job_text: str) -> np.ndarray:
    """One embedding for the whole job: mean of its chunk embeddings, re-normalised."""
    v = encode(chunk(job_text)).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


# ------------------------------------------------------------------ similarity
def resume_similarity(resume_texts: list[str], job_text: str) -> np.ndarray:
    """Raw cosine of every resume against the job, shape (n,).

    All chunks from all resumes are encoded in one batched call, which is much
    faster than encoding resume by resume.
    """
    if not resume_texts:
        return np.zeros(0, dtype=np.float32)
    jv = job_vector(job_text)

    all_chunks, owner = [], []
    for i, text in enumerate(resume_texts):
        for c in chunk(text):
            all_chunks.append(c)
            owner.append(i)
    sims = encode(all_chunks) @ jv
    owner = np.asarray(owner)

    out = np.zeros(len(resume_texts), dtype=np.float32)
    for i in range(len(resume_texts)):
        s = np.sort(sims[owner == i])[::-1]
        out[i] = float(s[:TOP_K_CHUNKS].mean()) if len(s) else 0.0
    return out


def calibrate(raw):
    """Map raw cosine onto the 0-1 score scale using SIM_FLOOR / SIM_CEIL."""
    return np.clip((np.asarray(raw, dtype=np.float32) - SIM_FLOOR) /
                   (SIM_CEIL - SIM_FLOOR), 0.0, 1.0)


# ------------------------------------------------------------------ related skills
def _skill_vectors(skills: list[str]) -> np.ndarray:
    new = [s for s in skills if s not in _skill_cache]
    if new:
        for s, v in zip(new, encode(new)):
            _skill_cache[s] = v
    return np.stack([_skill_cache[s] for s in skills])


def related_skills(missing: list[str], candidate_skills: list[str],
                   threshold: float = RELATED_SKILL_MIN) -> dict[str, tuple[str, float]]:
    """For each missing requirement, the candidate's most similar skill, if close enough.

    Example: the job needs PyTorch, the candidate lists TensorFlow -> related.
    Shown to the recruiter as a transferable skill; it does not change the score.
    """
    if not missing or not candidate_skills:
        return {}
    sims = _skill_vectors(missing) @ _skill_vectors(candidate_skills).T
    out = {}
    for i, skill in enumerate(missing):
        j = int(np.argmax(sims[i]))
        if sims[i, j] >= threshold and candidate_skills[j] != skill:
            out[skill] = (candidate_skills[j], round(float(sims[i, j]), 3))
    return out

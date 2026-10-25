"""Download Sentence-BERT once and verify it works.

    python tools/download_model.py

Needs internet the first time (~90 MB). The model is saved to ml/sbert_model/,
after which ResumeX runs fully offline, including during your demo.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SBERT_LOCAL_DIR, SBERT_MODEL_NAME  # noqa: E402


def main():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is not installed. Run:\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                 "  pip install -r requirements.txt")

    local = Path(SBERT_LOCAL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  |  device: {device}")

    t0 = time.time()
    if (local / "modules.json").exists():
        print(f"Found a saved model in {local}")
        model = SentenceTransformer(str(local), device=device)
    else:
        print(f"Downloading {SBERT_MODEL_NAME} ...")
        model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        local.mkdir(parents=True, exist_ok=True)
        model.save(str(local))
        print(f"Saved to {local}")
    print(f"Loaded in {time.time() - t0:.1f}s  |  "
          f"{model.get_sentence_embedding_dimension()}-dimensional embeddings")

    # sanity check: related texts must score higher than unrelated ones
    job = "Machine learning engineer with Python, deep learning and SQL experience"
    tests = [
        ("Built deep learning models in Python and PyTorch; wrote SQL pipelines", "should be HIGH"),
        ("Data analyst using SQL, Excel dashboards and some Python scripting", "should be MEDIUM"),
        ("Executive chef running a 40-cover kitchen, menu costing and food safety", "should be LOW"),
    ]
    emb = model.encode([job] + [t for t, _ in tests], normalize_embeddings=True)
    print("\nSimilarity to:", job)
    scores = []
    for (text, note), vec in zip(tests, emb[1:]):
        s = float(vec @ emb[0])
        scores.append(s)
        print(f"  {s:.3f}  {note:17} {text[:60]}")

    if scores[0] > scores[1] > scores[2]:
        print("\nSBERT is working correctly. Start ResumeX with:  python run.py")
    else:
        print("\nWarning: the ordering is not as expected. The model loaded, but check the install.")


if __name__ == "__main__":
    main()

# ResumeX — Explainable AI Resume Screening

Classical implementation phase of the Quantum-Enhanced Resume Screening and Skill
Matching System. Recruiters create a job, ResumeX extracts the requirements,
screens bulk resumes, ranks candidates and explains every rank with evidence from
the resume itself.

**Understand → Match → Rank → Explain**

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows    (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tools/download_model.py      # downloads + tests Sentence-BERT once

# copy resume_job_matches.csv and resumes.csv from the ResumeX dataset into data/
python ml/train_models.py           # ~20 seconds
python tools/make_test_resumes.py 25

python run.py                       # http://127.0.0.1:8000
```

Full instructions, including VS Code setup and troubleshooting, are in
**ResumeX_Build_Guide.docx**.

---

## What is implemented

| Feature | Where |
|---|---|
| Recruiter registration, login, dashboard | `app/routers/auth.py`, `jobs.py` |
| Job creation + AI requirement extraction | `app/services/jd_extractor.py` |
| Must Have / Preferred / Nice to Have categories | `app/models.py`, `job_requirements.html` |
| Recruiter-configurable weights and importance | `job_requirements.html`, `static/js/app.js` |
| Hard filters (experience, education) | `app/services/scoring.py` |
| Bulk PDF / DOCX / TXT upload | `app/routers/screening.py` |
| Resume parsing + section detection | `app/services/parser.py`, `extractor.py` |
| Skill normalisation (144 canonical skills) | `app/services/ontology.py` |
| Sentence-BERT semantic matching (chunked) | `app/services/embeddings.py` |
| SBERT related / transferable skills for gaps | `embeddings.related_skills` |
| Weighted hybrid scoring | `app/services/scoring.py` |
| Ranking + Top 10/20/50 shortlists | `app/routers/candidates.py`, `results.html` |
| Skill-gap analysis + counterfactual uplift | `scoring.uplift_if_added`, `candidate.html` |
| Explainable ranking with resume evidence | `scoring.explain`, `candidate.html` |
| Candidate comparison | `compare.html` |
| Recruitment stage tracking | `Candidate.stage`, `results.html` |
| Analytics dashboard | `app/routers/analytics.py` |
| SVM / Random Forest comparison | `ml/train_models.py` |
| Accuracy, Precision, Recall, F1, NDCG@10, MRR, time | `ml/evaluate.py` |

Quantum is **not** implemented in this phase, by design. The seam where it plugs
in next semester is `app/services/ml_models.py` — see section 12 of the guide.

---

## How the score is calculated

```
SCORE = w1·skills + w2·experience + w3·education + w4·semantic
        × 0.5 if the candidate evidences no must-have skill

skills      importance-weighted coverage of the job's requirements
            (Must Have 1.0, Preferred 0.6, Nice to Have 0.3)
experience  min(1, candidate years / required years)
education   1.0 if level ≥ required, else −0.34 per level short
semantic    Sentence-BERT cosine, mapped from 0.15–0.65 onto 0–1
```

Hard filters are checked separately. Candidates who fail them stay **visible** in
a "Filtered out" list with the reason shown — nothing is silently rejected.

The explanation on the candidate page is generated from this same arithmetic, so
it can never disagree with the ranking.

---

## Design choices worth defending in the viva

- **Rule-based extraction, not generative.** A language model asked to extract
  skills can invent a qualification that is not in the resume. A rule-based
  extractor can only fail to find one. Missing is a safe error; fabricated is not.
- **Explanations are computed, not written.** Every number comes from the scoring
  function itself.
- **Hidden text is stripped at parse time.** White or sub-4pt characters in a PDF
  are dropped before scoring — the standard resume manipulation trick.
- **No career-gap or job-hopping penalty.** These act as proxies for age and
  disability and are the subject of active litigation against commercial vendors.
- **Weights are the recruiter's.** Changing a slider re-scores instantly from
  stored parse results, without re-reading any file.

---

## Requirements

Python 3.11 or 3.12. Sentence-BERT is required: the server loads it at startup and
refuses to start with a clear message if it is missing. After
`python tools/download_model.py` the model is stored in `ml/sbert_model/` and the
app runs fully offline. Check the running model at `/api/status`.

## Deploying it

See **deploy/DEPLOY.md**. Two free routes with enough memory for PyTorch and
SBERT: Hugging Face Spaces (no credit card) and Google Cloud Run. The `Dockerfile`
works for both and bakes the SBERT model into the image so the app starts fast.

Run `python ml/train_models.py` before deploying so the trained models ship with
your code.

## Layout

```
app/routers/     one file per screen group
app/services/    the matching intelligence
ml/              training and evaluation scripts
templates/       15 HTML screens
static/          design system (CSS variables at the top of style.css)
tools/           test-resume exporter
data/            put the dataset CSVs here
```

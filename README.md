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
pip install -r requirements.txt     # includes the OCR engine for scanned resumes
python tools/check_ocr.py           # optional: confirms OCR works
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
| Bulk PDF / DOCX / TXT / image upload, OCR for scanned PDFs (`app/services/ocr.py`) | `app/routers/screening.py` |
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

## Scanned resumes (OCR) and extraction accuracy

Many real resumes are scans or images saved as PDF: there is no text inside, only
a picture of the page. ResumeX reads those pages with OCR (`app/services/ocr.py`).
Images (.png, .jpg, .tif) can be uploaded too.

* **RapidOCR** (default, `app/services/ocr_rapid.py`) is installed by
  `requirements.txt` and carries its models inside the package, so nothing else
  has to be downloaded or installed. Its recogniser sometimes drops spaces
  between English words; these are restored with a word-frequency splitter
  (`wordninja`), and the page is read column by column.
* **Tesseract** is used instead when it happens to be installed (it is a separate
  program; on Windows https://github.com/UB-Mannheim/tesseract/wiki). It is a
  little more accurate on this dataset. Force either with
  `RESUMEX_OCR_ENGINE=rapidocr` or `RESUMEX_OCR_ENGINE=tesseract`.

What the parser reads from each resume, and how:

| Field | How |
|---|---|
| Name | The biggest text near the top of page 1 that reads as a name (text size comes from the PDF fonts, or from OCR word boxes) |
| Job title | The role line under the name ("Senior Java Developer"), with junk around it removed |
| Years of experience | Union of dated job ranges: `Jan 2016 - Nov 2018`, `05/2018 - Present`, `2014-06 - 2016-05`, `2015 - Ongoing`; school and degree lines are skipped; "Present" means today |
| Degree | Bachelor / Master / ...; short forms like MS or BA only count when written as a degree, so "Boston, MA" or "Dear Ms." are not degrees |
| Skills | Ontology match, including the Java ecosystem (Spring, Spring Boot, Hibernate, JPA, Java EE, Microservices, REST API, Maven, JUnit, Kafka, Docker, Kubernetes, AWS, Git...). A specific skill also counts for its broader one (Docker -> DevOps) |
| Document type | Cover letters and job postings mixed into a bulk upload are flagged and moved to *Filtered out* |

Requirements typed by hand are matched leniently: "java", "J2EE", "springboot"
map to the ontology's names, and skills that are not in the ontology at all are
matched as a phrase in the resume text.

**Measured accuracy** against hand-checked answers for 38 of the 72 scanned
Java-developer resumes (`python tools/evaluate_extraction.py path/to/JavaDeveloper -v`).
30 resumes were used while improving the parser; 8 more were labelled afterwards,
as a check on resumes the rules were not tuned on. Each field is scored only where
it applies (a cover letter has no candidate name).

| Field | Before (no OCR) | RapidOCR (default), tuning 30 / held-out 8 | Tesseract, tuning 30 / held-out 8 |
|---|---|---|---|
| Name | 0% | 92% / 88% | 92% / 88% |
| Job title | 0% | 93% / 100% | 93% / 100% |
| Years of experience (within 1 year) | 7%* | 87% / 88% | 93% / 88% |
| Degree level | 30%* | 90% / 100% | 97% / 100% |
| Java found | 0% | 100% / 100% | 100% / 100% |
| Cover letter / job posting detected | - | 97% / 100% | 97% / 100% |

\* only where the true answer was "none": every scan was read as an empty page.
The held-out set was lower when first measured (Tesseract: names 5/8, titles,
years and degrees 7/8); fixes prompted by it raised it, so treat it as indicative.

The remaining misses are scans too small or blurred for OCR (300-500 pixel wide
images, where dates or digits are misread or skipped), and genuinely ambiguous
cases. They cannot be fixed in the parser; a better scan fixes them.

---

## Requirements

Python 3.11 or 3.12 (the built-in OCR engine supports up to 3.12). Tesseract is
optional; if installed somewhere unusual, set `TESSERACT_CMD` to the full path of
`tesseract.exe`. Sentence-BERT is required: the server loads it at startup and
refuses to start with a clear message if it is missing. After
`python tools/download_model.py` the model is stored in `ml/sbert_model/` and the
app runs fully offline. Check the running model at `/api/status`.

## Deploying it

**Railway (from GitHub): see deploy/RAILWAY.md.** Push this folder to a GitHub
repository, create a Railway project from it, set `RESUMEX_SECRET`, attach a
volume at `/data` with `RESUMEX_STATE_DIR=/data`, and generate a domain.

Other routes are in **deploy/DEPLOY.md**: Hugging Face Spaces (no credit card)
and Google Cloud Run. The same `Dockerfile` works for all of them and bakes the
SBERT model into the image so the app starts fast.

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

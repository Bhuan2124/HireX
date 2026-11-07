# HireX: Explainable AI Hiring, for Both Sides

HireX (formerly ResumeX) is a two-sided hiring platform. **Recruiters** post jobs
with openings and a deadline; HireX reads every resume, ranks the applicants and
explains each rank with evidence. **Candidates** search jobs, check how well they
match *before* applying, and after the deadline see their final rank and a
"Why not me?" explanation built from the same score the recruiter sees.

Classical implementation phase of the Quantum-Enhanced Resume Screening and Skill
Matching System (quantum comes next semester).

**Understand → Match → Rank → Explain**, for recruiters and candidates.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows    (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt     # includes the OCR engine for scanned resumes
python tools/download_model.py      # downloads + tests Sentence-BERT once

python tools/seed_demo.py --reset   # demo recruiter, 6 candidates, 3 jobs (password demo123)
python run.py                       # http://127.0.0.1:8000

python tests/test_practical_scoring.py   # optional: checks internship / project scoring
```

Optional, as before: copy the dataset CSVs into `data/` and run
`python ml/train_models.py` to add the SVM / Random Forest relevance model and the
field classifier (the classifier makes "Not eligible" smarter: it can tell that a
fitness resume is from a different field).

Setup details and troubleshooting: **docs/HireX_Build_Guide_setup.docx** (written for
ResumeX v3; installation steps are unchanged).

### Demo walkthrough (after `seed_demo.py`)

| Log in as | Password | Try this |
|---|---|---|
| `hr@acme.com` (recruiter) | demo123 | Jobs → *Java Developer (Freshers)* → ranking with the interview-zone line, a "Possible copy" flag, decisions |
| `sneha@demo.com` (candidate) | demo123 | My applications → Java job → final rank, "Why not me?", recruiter's reason |
| `ananya@demo.com` (candidate) | demo123 | Find jobs → *Backend Engineer* → Check my match → Apply |
| `priya@demo.com` (fitness coach) | demo123 | Find jobs → *Backend Engineer* → Check my match → **Not eligible** warning |

---

## What is new in HireX (v4)

| Feature | Where |
|---|---|
| Two account types: recruiter and candidate | `app/models.py` (`CandidateUser`), `routers/auth.py`, `security.py` |
| Job posting with openings, application deadline, Fresher / Experienced / Any, interview zone | `models.Job`, `job_new.html`, `job_requirements.html` |
| Candidate job search (role or skill, city, field, type, level) | `routers/portal.py` → `/find-jobs` |
| **Check my match** before applying: %, label, found / missing skills, **Not eligible** warning (can still apply) | `portal.check_match`, `match_result.html`, `scoring.fit_label` |
| One application per job, **no resubmission** | `portal.apply` |
| **Rank hidden until the deadline**, then final; results released on the first page load after it | `services/platform.release_due_jobs` |
| **Why not me?** final rank, interview zone, places away, where you were behind the zone, which missing skill would have moved you up and to which rank | `platform.feedback`, `application.html` |
| Recruiter decisions shown to the candidate (Interview with date, Selected, Not selected with reason + note), only after the deadline | `platform.record_decision`, `candidate_status` |
| **Practical work score: internships + projects** (for freshers) | `services/practical.py` |
| Bulk actions: shortlist the interview zone, reject the rest with a reason | `routers/candidates.py` |
| Duplicate / copied resume detection | `platform.flag_duplicate` |
| Notifications (bell) for both sides | `routers/notifications.py` |
| Recruiter bulk upload kept for resumes received elsewhere (never deletes HireX applications) | `routers/screening.py` |

Everything from v3 is still there: AI requirement extraction, Must Have / Preferred /
Nice to Have, weights, hard filters, OCR, SBERT matching, evidence, comparison,
analytics, SVM / Random Forest evaluation.

---

## How the score is calculated

```
SCORE = w1·skills + w2·experience + w3·practical + w4·education + w5·semantic
        × 0.5 if the candidate evidences no must-have skill

skills      importance-weighted coverage of the job's requirements
            (Must Have 1.0, Preferred 0.6, Nice to Have 0.3)
experience  min(1, candidate years / required years)   (fresher jobs: not used)
practical   internships + projects, see below
education   1.0 if level ≥ required, else −0.34 per level short
semantic    Sentence-BERT cosine, mapped from 0.15–0.65 onto 0–1
```

Default weights by job type (the recruiter can change them with the sliders):

| | Skills | Experience | Practical work | Education | Overall fit |
|---|---|---|---|---|---|
| Fresher | 35% | 0% | **30%** | 15% | 20% |
| Experienced | 40% | 25% | 0% | 15% | 20% |
| Any | 35% | 15% | 15% | 15% | 20% |

### Practical work: internships and projects (`app/services/practical.py`)

A fresher has no full-time job, so Experience is 0 for every fresher. Without this
part, a student with two relevant internships and three strong projects ranks the
same as a student with nothing.

```
Internships   duration  = min(1, total months / 6)
              relevance = job skills used in the internships (70%) + SBERT similarity (30%)
              score     = 0.35 × duration + 0.65 × relevance
              (no internship but a real job? the job counts here instead)

Each project  relevance = job skills used in that project (70%) + SBERT similarity (30%)
              depth     = 0.4 × description length + 0.3 × GitHub/live link + 0.3 × measurable result
              score     = 0.75 × relevance + 0.25 × depth
Projects      = 0.60 × best + 0.25 × 2nd best + 0.15 × 3rd best
              (an extra weak project can only add points, never lower the score)

Practical     both present → 0.5 × internships + 0.5 × projects
              only one     → 0.8 × that one   (no internship? projects still earn up to 80%)
```

Only what is written counts: an internship without dates counts as 2 months, a
project with no description earns little, a skill counts only if it appears inside
that internship or project.

### Eligibility and the interview zone

* **Not eligible** = fails a hard filter (experience, education), or has no must-have
  skill at all *and* is from a different field / has almost nothing in common with
  the job. It is a warning: the candidate may still apply, and is ranked below every
  eligible applicant.
* **Interview zone** = the top N eligible applicants (N = 3 × openings unless the
  recruiter sets it). After the deadline every candidate sees whether they were in it
  and how many places away.

Hard filters are checked separately. Candidates who fail them stay **visible** in the
"Not eligible" list with the reason shown. Nothing is silently rejected.

The explanation on both sides is generated from this same arithmetic, so it can
never disagree with the ranking.

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
- **Ranks are hidden until the deadline.** Before it, a rank changes with every new
  applicant and would mislead. After it, the ranking is final and fully explained.
- **"Why not me?" never names other applicants.** It compares the candidate with the
  interview zone's *average*, and estimates the rank a missing skill would have given.
- **"Not eligible" warns, it does not block.** The parser can misread a scan; the
  candidate decides.
- **Copies are found by wording, not meaning.** SBERT would call two genuine Java
  developers "similar". Shared 5-word phrases (Jaccard ≥ 0.60) catch copied resumes
  without flagging people who simply have the same skills.
- **Freshers are scored on what they built.** Experience weight moves to practical
  work for fresher jobs; projects alone can still earn 80% of it, so students who
  couldn't get an internship are not shut out.

---

## Scanned resumes (OCR) and extraction accuracy

Many real resumes are scans or images saved as PDF: there is no text inside, only
a picture of the page. HireX reads those pages with OCR (`app/services/ocr.py`).
Images (.png, .jpg, .tif) can be uploaded too.

* **RapidOCR** (default, `app/services/ocr_rapid.py`) is installed by
  `requirements.txt` and carries its models inside the package, so nothing else
  has to be downloaded or installed. Its recogniser sometimes drops spaces
  between English words; these are restored with a word-frequency splitter
  (`wordninja`), and the page is read column by column.
* **Tesseract** is used instead when it happens to be installed (it is a separate
  program; on Windows https://github.com/UB-Mannheim/tesseract/wiki). It is a
  little more accurate on this dataset. Force either with
  `HIREX_OCR_ENGINE=rapidocr` or `HIREX_OCR_ENGINE=tesseract`.

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
repository, create a Railway project from it, set `HIREX_SECRET`, attach a
volume at `/data` with `HIREX_STATE_DIR=/data`, optionally `HIREX_SEED_DEMO=1`
for the demo accounts, and generate a domain. `railway.json` sets the health check.

Other routes are in **deploy/DEPLOY.md**: Hugging Face Spaces (no credit card)
and Google Cloud Run. The same `Dockerfile` works for all of them and bakes the
SBERT model into the image so the app starts fast.

## Layout

```
app/routers/
  auth.py            register / log in, recruiter or candidate
  jobs.py            recruiter: post a job, requirements, weights, deadline
  candidates.py      recruiter: ranking, candidate page, decisions, compare, search
  screening.py       recruiter: bulk upload of resumes received elsewhere
  analytics.py       recruiter: funnel, score distribution, skill gaps
  portal.py          candidate: find jobs, check my match, apply, my applications
  notifications.py   both: the bell
app/services/
  parser.py, extractor.py, ocr*.py   read resumes (text, OCR, sections, dates)
  practical.py       internships + projects (new)
  analysis.py        score one resume against one job (shared by every path, new)
  scoring.py         the formula, labels and explanation
  ranking.py         store applications, bulk screening, rank, re-score
  platform.py        deadlines, decisions, "Why not me?", duplicates, notifications (new)
  embeddings.py      Sentence-BERT
ml/                  training and evaluation scripts
templates/           22 HTML screens (recruiter + candidate)
static/              design system (CSS variables at the top of style.css)
tools/seed_demo.py   demo data for reviews; tools/demo_resumes/ holds its resumes
tests/               python tests/test_practical_scoring.py
data/                put the dataset CSVs here
```

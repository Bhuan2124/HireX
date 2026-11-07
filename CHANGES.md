# HireX v4: two-sided platform (from ResumeX v3)

## Renamed
ResumeX is now **HireX**: the product serves recruiters *and* candidates.
Environment variables are now `HIREX_*` (was `RESUMEX_*`) and the database file is
`hirex.db`. Run `python tools/seed_demo.py --reset` for a fresh demo database.

## New: candidate side
- Candidate accounts, job search with filters, job pages showing the requirements.
- **Check my match** before applying: match %, Strong / Good / Low / Weak match or
  **Not eligible** (warning only), found and missing skills, how the % adds up.
- One application per job, no resubmission.
- **My applications**: before the deadline, match % and "results after the deadline";
  after it, final rank, interview zone, decision, recruiter's reason and note, and
  **Why not me?**.

## New: recruiter side
- Openings, application deadline, job level (Fresher / Experienced / Any) and
  interview-zone size on every job.
- Interview-zone line on the ranking, "Shortlist top N", "Reject the rest" with a reason.
- Decision form: Interview (date and time), Selected, Not selected (reason + note).
  Candidates see decisions only after the deadline.
- "Possible copy" flag for near-duplicate resumes.
- Bulk upload now only replaces the previous bulk upload; HireX applications are kept.
  (It also now deletes evidence rows with the candidates it replaces.)

## New: practical work score (internships + projects)
- `app/services/practical.py` finds internships (an Internships section, or "Intern"
  entries inside Experience) with their months, and splits the Projects section into
  projects (titles, numbered lists, the dataset's one-line layout).
- Fifth weight, **Practical work**. Fresher jobs default to 30% practical, 0% experience.
- Shown on the recruiter's candidate page and in the candidate's feedback.
- The v3 `project_score` (SBERT similarity of projects + experience) was computed but
  never used in the total; it is replaced by the projects part of practical work.

## Fixes found while building
- "Not eligible" applicants (no must-have skill, different field) could fill the
  interview zone when better applicants failed a hard filter. Ranking now puts every
  eligible applicant first.
- "What would help" estimates now recompute the full score, so adding a missing
  must-have also lifts the 50% "no must-have" penalty.
- All timestamps use the server's local time, matching the deadline.

---

# Changes in this version

## The 20% bug

All 72 Java-developer resumes are scanned images saved as PDF: there is no text
inside them. The parser only read PDF text, so every resume came out empty: no
Java, 0 years, no degree. The score was then the same for everyone:

    skills 0 + experience 25% (no minimum set) + education 15% (no minimum set) + semantic 0
    = 40%, halved by the "no must-have skill" rule = 20%

## Reading resumes

- **OCR for scanned PDFs and images** (`app/services/ocr.py`), with nothing to
  install beyond `requirements.txt`: the default engine is RapidOCR
  (`app/services/ocr_rapid.py`, models inside the pip package). It reads the page
  column by column (sidebars) and restores spaces its recogniser drops between
  English words. If the Tesseract program is installed it is used instead (a
  little more accurate); small scans are then enlarged with a smooth filter, and
  a second layout mode is tried when Tesseract skips part of a page.
  `.png/.jpg/.tif/.bmp/.webp` uploads are accepted too.
- **No silent zero scores.** A file with no readable text is listed as an error
  on the processing page instead of being ranked as an empty resume. If no OCR
  engine is available, one short message says what to run.
- **Resumes are parsed in parallel** (`HIREX_PARSE_WORKERS`, default up to 4),
  because OCR takes 2-4 seconds a page.

## Extraction accuracy

- **Name and title** come from text size (the biggest text near the top), using
  PDF font sizes or OCR word heights. Contact lines, places, section headings and
  OCR noise are rejected. Letter-spaced names split by OCR ("Ro bert S m ith")
  are joined back together.
- **Years of experience**:
  - New date formats: `2014-06 - 2016-05`, `2015 - Ongoing`, `Dec 2014 - 2020`
    and `05-2018`. Two-column layouts that put the job title between the two
    dates are also handled.
  - "Present" now means today. Before, it meant January 2020, so any job that
    started after 2020 was thrown away.
  - School and degree dates are not counted as work.
- **Degree**: "Boston, MA", "Dear Ms.", "MS-SQL" and "Scrum Master" no longer
  count as master's degrees. "Bachelors", "Masters'", "BSc" and "BS Computer
  Science" are now recognised.
- **Java ecosystem skills** were added: Java EE, Spring, Spring Boot,
  Hibernate, JPA, JDBC, JSP/Servlets, Struts, Microservices, REST API, Maven,
  Gradle, JUnit, Mockito, Multithreading, OOP, Design Patterns, Kafka, JMS,
  Tomcat, Application Servers, Docker, Kubernetes, Jenkins, AWS, Git and MongoDB.
  A specific skill still counts for its old broader skill (Docker -> DevOps,
  AWS -> Cloud Computing), so existing jobs and trained models behave as before.
  "Spring" only counts as a skill near another tech word, so an address like
  "Denise Spring, Houston" is not read as the Spring framework.
- **Cover letters and job postings** in a bulk upload are detected and moved
  to *Filtered out* with the reason shown.

Measured accuracy is in the README. Run it yourself with
`python tools/evaluate_extraction.py path/to/JavaDeveloper -v`.

## Bugs fixed

- Requirements added by hand were stored exactly as typed ("java", "spring
  boot") but compared with canonical names ("Java"), so they showed as missing
  for every candidate. They are now mapped to the canonical skill. Skills that
  are not in the ontology at all are matched as a phrase in the resume.
- Changing the stage of a candidate that does not exist crashed the server.
- Candidates with equal scores were in arbitrary order. They are now ordered by
  semantic similarity, then skill coverage, then experience.
- Files that could not be read during an otherwise successful screening were
  silently dropped. The processing page now lists them.

## Setup change

Just `pip install -r requirements.txt` (Python 3.11 or 3.12). Installing the
Tesseract program is optional; the Dockerfile includes it.
No database changes were made: existing `hirex.db` files keep working. Re-run
screening on a job to re-read its resumes with the new parser.

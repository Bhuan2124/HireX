# Deploying ResumeX on Railway (from GitHub)

Railway builds the `Dockerfile` in this folder automatically. Every `git push`
to GitHub redeploys the app. Allow about 20 minutes the first time.

**Plan:** ResumeX needs about 1 GB of RAM (PyTorch + Sentence-BERT). The
Railway **Trial** plan (one-time $5 credit, 1 GB RAM) is just enough. The
**Free** plan (0.5 GB) is too small. **Hobby** ($5/month) has plenty.

---

## 1. Put the code on GitHub

1. Go to https://github.com/new
   - Repository name: `resumex`
   - Private or Public, either works
   - Do **not** tick "Add a README" (the project already has one)
   - Click **Create repository**
2. On your laptop, open a terminal **inside the project folder** (the one that
   contains `Dockerfile`, `requirements.txt` and the `app` folder), then run
   (replace `YOUR-USERNAME`):

   ```
   git init
   git add .
   git commit -m "ResumeX"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/resumex.git
   git push -u origin main
   ```

   If Git asks you to sign in, use your GitHub account. If it says
   `git is not recognized`, install Git from https://git-scm.com/download/win
   and open a new terminal.

3. Refresh the GitHub page. You should see `Dockerfile`, `app/`, `templates/`
   and the other files at the top level of the repository (not inside an extra
   folder).

> Trained models: if you ran `python ml/train_models.py` on your laptop, the
> files `ml/models/relevance_model.joblib` and `ml/models/category_model.joblib`
> are pushed with the code and the deployed app uses them. Without them the app
> still works; only the ML relevance column is empty.

## 2. Create the Railway project

1. Go to https://railway.com and **log in with GitHub**.
2. Click **New Project**, then **Deploy from GitHub repo**, then pick `resumex`.
   If the repository is not listed, click **Configure GitHub App** and give
   Railway access to it.
3. Railway finds the `Dockerfile` and starts building. The first build takes
   10-15 minutes, because it installs PyTorch and downloads the Sentence-BERT
   model into the image. You can watch it under **Deployments, then View logs**.

## 3. Settings (do these while it builds)

Click the `resumex` service, then:

**a. Variables tab**, then **New Variable**:

| Name | Value |
|---|---|
| `RESUMEX_SECRET` | a long random string (see below) |
| `RESUMEX_STATE_DIR` | `/data` |

Make the random string on your laptop with:
```
python -c "import secrets; print(secrets.token_hex(32))"
```
Without `RESUMEX_SECRET`, everyone is logged out whenever the app restarts.

**b. Add a volume**, so accounts, jobs and candidates survive redeploys.
Right-click the service on the project canvas (or press Ctrl+K), choose
**Add Volume**, and set the mount path to `/data`. Without a volume,
everything is wiped on each redeploy, because the container's disk is temporary.

**c. Settings tab, then Networking**: click **Generate Domain**. If it asks for
a port, enter `8080`. This gives you the public link, like
`https://resumex-production.up.railway.app`.

**d. Settings tab, then Deploy** (optional): set **Healthcheck Path** to
`/api/status`, so a broken deploy never replaces a working one.

After changing variables or adding the volume, Railway redeploys by itself.

## 4. Check it works

- Open `https://YOUR-DOMAIN/api/status`. You should see `"sbert": {... "loaded": true ...}`
  and `"ocr": {"available": true, "label": "Tesseract 5..."}`.
- Open `https://YOUR-DOMAIN`, register, create a job, and upload a few resumes.
  Scanned resumes take a few seconds each.

## Updating the app later

Change the code on your laptop, then:
```
git add .
git commit -m "describe the change"
git push
```
Railway rebuilds and redeploys automatically. Thanks to the volume, your data stays.

## Optional: Postgres instead of the volume

On the project canvas: **New, then Database, then PostgreSQL**. In the `resumex` service's
Variables add `DATABASE_URL` with the value `${{Postgres.DATABASE_URL}}`. Keep
the `/data` volume anyway: uploaded resume files are stored there.

## If something goes wrong

| What you see | What to do |
|---|---|
| Build log stops at `pip install` with "No matching distribution" | Make sure the latest `requirements.txt` was pushed (`git push`). |
| Deploy crashes with "Out of memory" / exit code 137 | The plan has too little RAM. Free (0.5 GB) cannot run it; use Trial or Hobby. |
| Page says "Application failed to respond" | Wait 1-2 minutes after the deploy finishes (the model loads at start). Check that the domain's port is `8080`, or remove the port so Railway uses `PORT`. |
| Logged out after every deploy | Set `RESUMEX_SECRET` (step 3a). |
| Jobs disappear after a deploy | Add the volume at `/data` and set `RESUMEX_STATE_DIR=/data` (step 3a/3b). |
| Screening a large batch is slow | Normal on 1-2 vCPU: OCR takes 2-5 seconds per scanned page. |

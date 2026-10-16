# Deploying ResumeX for free

Your app needs about **1 GB of RAM** because it loads PyTorch and Sentence-BERT.
That single fact rules out most free tiers, which give 512 MB. The two routes
below have enough memory.

**Before anything else**, run this once on your laptop so the trained models are
in the folder you deploy:

```
python ml/train_models.py
```

It writes `ml/models/relevance_model.joblib` and `ml/models/category_model.joblib`.
Both are committed with your code (the `.gitignore` already allows them).

---

## Route A — Hugging Face Spaces (no credit card, ~10 minutes)

Try this first. If your account can't create a CPU Space, go to Route B.

**1. Create the Space**

Go to https://huggingface.co/new-space and set:

- Owner: your username
- Space name: `resumex`
- License: `mit`
- **Space SDK: Docker** → pick **Blank**
- **Space hardware: CPU basic (free)**
- Visibility: Public

If Docker or CPU basic is greyed out or marked as paid, your account can't do
this. Skip to Route B.

**2. Swap in the Space README**

Hugging Face reads settings from the top of `README.md`. From your project folder:

```
copy deploy\README_huggingface.md README.md
```

(macOS/Linux: `cp deploy/README_huggingface.md README.md`)

**3. Push your code**

```
git init
git add .
git commit -m "ResumeX"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/resumex
git push space main
```

Replace `YOUR_USERNAME`. When git asks for a password, paste an access token from
https://huggingface.co/settings/tokens (create one with **write** permission).

**4. Wait for the build**

Open your Space page and watch the **Logs** tab. The first build takes 10–15
minutes because it downloads PyTorch and the SBERT model. When it finishes you'll
see `[SBERT] all-MiniLM-L6-v2 ready on cpu` and the app appears.

Your URL: `https://huggingface.co/spaces/YOUR_USERNAME/resumex`

---

## Route B — Google Cloud Run (needs a card for verification, never charges)

Google asks for a card to prove you're a person. The always-free tier covers
2 million requests a month; a demo uses a handful. You also get $300 of trial
credit. Set a budget alert if you want to be certain.

**1. Install the Google Cloud CLI**

Download from https://cloud.google.com/sdk/docs/install and run the installer.
Then close and reopen your terminal.

**2. Log in and create a project**

```
gcloud auth login
gcloud projects create resumex-demo-001 --name="ResumeX"
gcloud config set project resumex-demo-001
```

If the project name is taken, add different digits.

**3. Link billing**

Open https://console.cloud.google.com/billing, add your card, then attach it to
the `resumex-demo-001` project. This is the only step that needs the browser.

**4. Turn on the services**

```
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

**5. Deploy**

From your project folder (the one with the `Dockerfile`):

```
gcloud run deploy resumex --source . --region asia-south1 --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 600 --set-env-vars RESUMEX_SECRET=change-me-to-something-random
```

Say **yes** when it offers to create an Artifact Registry repository.

The first deploy takes 10–15 minutes while it builds the image. When it's done it
prints your URL, something like
`https://resumex-xxxxxxxx-el.a.run.app`. That's your live app.

`asia-south1` is Mumbai. Use `us-central1` if you'd rather.

**Redeploying after a code change:** run the same `gcloud run deploy` command again.

---

## If you get stuck, here's the demo-day fallback

This gives you a public HTTPS link to the app running on your own laptop, in
about two minutes, free and with no account.

1. Download `cloudflared` from
   https://github.com/cloudflare/cloudflared/releases (the `windows-amd64.exe`).
2. Start ResumeX normally: `python run.py`
3. In a second terminal, in the folder with cloudflared:

```
cloudflared tunnel --url http://localhost:8000
```

It prints a `https://something-random.trycloudflare.com` URL that anyone can open
while your laptop stays on. Not a real deployment, but your demo works over the
internet.

---

## Things to know about the deployed app

**Data doesn't survive restarts.** Jobs and uploaded resumes live in temporary
storage, so they're cleared when the container restarts or sleeps. For a demo this
is fine: you create a job and upload resumes live. If your mentor wants the data to
persist, create a free Postgres at https://neon.tech, copy its connection string,
and set it as an environment variable:

```
DATABASE_URL=postgresql://user:password@host/dbname
```

Add `psycopg2-binary==2.9.10` to `requirements.txt` and redeploy. Nothing else
changes — SQLAlchemy handles the rest.

**The first request after idle is slow.** Cloud Run scales to zero and Spaces
sleep, so the first visit after a quiet period takes 20–40 seconds to wake up.
Open the app a minute before you demo it.

**Check that it's healthy** by visiting `/api/status` on your deployed URL. It
returns which model is loaded and on what device.

**Keep the dataset out of the repo.** The `.gitignore` excludes `data/*.csv`.
The deployed app doesn't need those files; only the trained `.joblib` models.

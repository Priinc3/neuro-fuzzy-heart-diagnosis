# Deploy to Streamlit Community Cloud

The app is public on GitHub at <https://github.com/Priinc3/neuro-fuzzy-heart-diagnosis>.
Streamlit Community Cloud needs *your* browser login, so the deploy itself is a
manual step. It takes about 3 minutes.

## Steps (do these once)

1. Open <https://share.streamlit.io> and **sign in with GitHub** (account `Priinc3`).
2. Click **"Create app"** (top-right) → **"Deploy a public app from GitHub"**.
3. Fill in:
   - **Repository:** `Priinc3/neuro-fuzzy-heart-diagnosis`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **App URL:** set the subdomain to `neuro-fuzzy-heart`
   → final URL `https://neuro-fuzzy-heart.streamlit.app`
5. **Important — before clicking Deploy:** open **"Advanced settings"** and choose
   **Python 3.12**.
   - `requirements.txt` is pinned and tested on Python 3.12.
   - The Python version **cannot be changed after deploy** without deleting and
     recreating the app:
     <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python>
6. Leave **Secrets** empty — the app needs no secrets, no API keys, no `.env`.
7. Click **Deploy**. First build takes ~5 minutes (deps install from the pinned
   `requirements.txt`). When the page shows the five tabs
   (🩺 Diagnose · 📊 Model comparison · 🧠 Inside the ANFIS · 🔵 Fuzzy patient
   clusters · ℹ️ About), it is live.

## After deploy

- Verify the URL <https://neuro-fuzzy-heart.streamlit.app> opens the Diagnose tab.
- Take screenshots for `docs/screens/*.png` (see "Screenshots" in
  `docs/PROJECT_REPORT.md`) and drop them into that folder, then commit.
- **Apps sleep after ~12 h of inactivity.** Open the app ~10 minutes before the
  demo so the cold start is over:
  <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>

## If something fails

- **Build error in logs** → check that Branch is `main` and Main file is `app.py`.
- **Dependency error** → `requirements.txt` is pinned for Python 3.12; deploying
  with 3.11/3.13 will break. Delete the app, redeploy with 3.12.
- **Repo not listed** → make sure you are signed in as `Priinc3` and the app has
  access to public repos (Community Cloud is free, no billing).

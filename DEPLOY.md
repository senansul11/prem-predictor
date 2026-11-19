# Putting the app online (free): Streamlit Community Cloud

Streamlit Community Cloud hosts the app straight from this GitHub repo, for free.
A daily GitHub Action refreshes the prediction files in `data/outputs/`; when that commit lands, the
hosted app redeploys itself. No Docker and no paid plan needed.

## 1. Add your API key to GitHub (once)
The daily refresh needs a footballdata.io API key (free at footballdata.io). Add it as a secret from
your own terminal, so it is never pasted anywhere else. Run this inside your clone of the repo and
paste the key when asked (nothing is shown as you paste):

```bash
gh secret set FOOTBALLDATA_IO_API_KEY
```

## 2. Push the code to GitHub
Push the repo to your own GitHub account. (Pushing also switches on the daily refresh.)

## 3. Deploy on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and **Continue with GitHub**. Authorise Streamlit.
   For a **private** repo, Streamlit asks for extra permission to read your private repos. Allow it.
2. Click **Create app** and choose **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-github-username>/prem-predictor`
   - **Branch:** `main`
   - **Main file path:** `app/Home.py`
   - **App URL:** choose a name, e.g. `prem-predictor`
4. Open **Advanced settings** and set **Python version** to **3.12**. No secrets are needed.
5. Click **Deploy**. The first build takes a few minutes.

Your app will be at `https://<name>.streamlit.app`. Share that link.

## 4. How it stays up to date
- Every morning (07:00 UTC) the **refresh** workflow downloads new results and fixtures, runs the
  data checks, re-runs the predictions and commits the new files.
- Streamlit notices the new commit and updates the app, usually within a minute.
- You can run the refresh by hand from GitHub: **Actions -> refresh -> Run workflow**
  (tick "Re-run even if nothing changed" to force it).

## Good to know
- **Sleeping app:** on the free plan the app goes to sleep after about 12 hours with no visitors.
  The next visitor sees a "wake this app up" button; it is back in around half a minute.
- **Private repo:** Streamlit's free plan allows a limited number of private apps (reportedly one).
  If it refuses, you can make the repo public: it holds no secrets (your key lives only in GitHub
  Secrets and your local `.env`).
- **Memory:** the free plan is limited, but this app is small (under 1 MB of data).

## If something goes wrong
- **A GitHub issue titled "Daily refresh failed" appears:** a data check failed. The app keeps showing
  the last good predictions. Open the linked run to see why (for example a new club whose name is
  not in `src/plpredictor/data/team_names.py`).
- **The refresh run fails at "Check the API key is set":** the secret from step 1 is missing.
- **Streamlit shows an error after deploying:** open the app's **Manage app** panel for the log.
- **Fixtures stop updating:** the free footballdata.io plan allows 1,000 requests a month; the daily
  refresh uses about 2 a day.
- **A new season starts (August):** the first refresh may fail until the new CSV is published and any
  newly promoted clubs are added to `team_names.py`.

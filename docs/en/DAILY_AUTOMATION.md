# Daily Auto-Generated Songs (Claude → GitHub → Kaggle → Google Drive)

A hands-off pipeline: every morning at **07:00 IST**, Claude writes fresh lyrics,
commits them to this repo, a **Kaggle 2×T4** kernel generates the song, and the
finished audio lands in a **Google Drive folder**.

```
  Daily 07:00 IST  (cron 30 1 * * *  UTC)
        │
        ▼
  Claude Code Routine  (/schedule, runs in this repo)
    • uses the acestep-songwriting skill to write lyrics
    • writes kaggle/songs.json
    • git commit + push  →  main
        │  (push touches kaggle/songs.json)
        ▼
  GitHub Action  .github/workflows/kaggle-trigger.yml
    • kaggle kernels push   →  creates a new kernel version AND runs it
    • secrets: KAGGLE_USERNAME, KAGGLE_KEY
        │
        ▼
  Kaggle kernel  (2×T4, internet ON)
    • git clone PRIVATE repo (main, via GITHUB_TOKEN)
    • reads kaggle/songs.json   (INPUT_MODE="file")
    • acestep-api generates the song(s)
    • uploads mp3 + manifest.json → Google Drive folder
    • secrets: GITHUB_TOKEN, GDRIVE_OAUTH_TOKEN, GDRIVE_FOLDER_ID
        │
        ▼
  Your Google Drive folder  ← finished songs appear here daily
```

Why the GitHub Action (not the routine) runs `kaggle kernels push`: a `/schedule`
routine sandbox has no secret store for Kaggle credentials, but it can push to the
connected repo. Keeping the Kaggle token as a GitHub secret keeps the routine
simple — it only writes lyrics and commits.

---

## What's in the repo

| File | Role |
|------|------|
| `kaggle/build_notebook.py` | Generates the single-cell kernel notebook. Defaults are now `INPUT_MODE="file"`, `OUTPUT_MODE="gdrive"`. |
| `kaggle/acestep_kaggle_2xT4_weekly.ipynb` | The kernel notebook — **committed** to the repo. It's generated from `build_notebook.py`; if you edit the builder, run it and commit the regenerated `.ipynb`. |
| `kaggle/songs.json` | **The daily artifact.** The routine overwrites this; the kernel reads it. |
| `kaggle/songs.example.json` | Schema documentation (not read by the kernel). |
| `kaggle/kernel-metadata.json` | Required by `kaggle kernels push`. Put your Kaggle username in `id`. |
| `.github/workflows/kaggle-trigger.yml` | Trigger: on push to `kaggle/songs.json`, pushes+runs the kernel. |
| `scripts/gdrive_oauth_setup.py` | One-time local helper to mint the Drive OAuth token. |

---

## One-time setup

Do these once. Afterwards the pipeline runs itself.

### 1. Google Drive refresh token (for personal "My Drive")

Service accounts **cannot** write to a personal Drive (`storageQuotaExceeded`), so
the kernel uploads as *you* using an **OAuth refresh token**. A refresh token needs
three values together — `client_id`, `client_secret`, `refresh_token` — and the
kernel accepts them either bundled in one secret or supplied as three. Pick one
method below. Both first require an OAuth client:

**Create the OAuth client (both methods need this)**
1. [Google Cloud Console](https://console.cloud.google.com/) → create/pick a project.
2. **APIs & Services → Library →** enable **Google Drive API**.
3. **OAuth consent screen →** add your own Google account under **Test users**.
4. **Credentials → Create credentials → OAuth client ID**.

**Method A — Google OAuth Playground (quickest, no script):**
1. Create the client as type **Web application** and add
   `https://developers.google.com/oauthplayground` as an **Authorized redirect URI**.
   Note its **Client ID** and **Client secret**.
2. Open the [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/) →
   gear icon (⚙) → tick **Use your own OAuth credentials** → paste client id/secret.
3. In "Step 1", enter the scope `https://www.googleapis.com/auth/drive` →
   **Authorize APIs** → approve.
4. "Step 2" → **Exchange authorization code for tokens** → copy the **Refresh token**.
5. You now have all three values. Set them as **three Kaggle secrets**:
   `GDRIVE_REFRESH_TOKEN`, `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`.

**Method B — local helper script (one bundled secret):**
1. Create the client as type **Desktop app** → download the JSON.
2. Locally:
   ```bash
   pip install google-auth-oauthlib google-auth
   python scripts/gdrive_oauth_setup.py path/to/client_secret.json
   ```
   A browser opens; approve. The script prints an **authorized-user JSON** (it
   contains the refresh token) — set it as the single Kaggle secret `GDRIVE_OAUTH_TOKEN`.

**Then (both methods):** in Google Drive, create/open the destination folder. Its
**folder id** is the part after `/folders/` in the URL → Kaggle secret `GDRIVE_FOLDER_ID`.

### 1b. GitHub token (the repo is PRIVATE)

The pipeline lives in the **private** repo `hardik2015/ace-step-1.5-private`, so
the Kaggle kernel needs a token to clone it.

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. **Resource owner** = your account; **Repository access** = *Only select
   repositories* → `ace-step-1.5-private`; **Permissions → Repository → Contents
   = Read-only**. Generate and copy the token (`github_pat_...`).
3. You'll add it as the Kaggle secret `GITHUB_TOKEN` in step 2.

### 2. Kaggle kernel (one-time UI config)

`kernel-metadata.json` cannot select the T4×2 accelerator or attach secrets — set
those in the UI once; they persist across every `kaggle kernels push`.

1. `kaggle/kernel-metadata.json` is already set to `allinone2015/acestep-daily`.
2. Push the kernel the first time so it exists (the notebook is already committed
   in `kaggle/`; only rebuild it if you changed `build_notebook.py`):
   ```bash
   pip install kaggle            # configure ~/.kaggle/kaggle.json with your token
   # python kaggle/build_notebook.py   # only if you edited the builder
   kaggle kernels push -p kaggle/
   ```
3. Open the kernel on kaggle.com → **Settings/Add-ons**:
   - **Accelerator = GPU T4 ×2**
   - **Internet = ON**
   - **Secrets →** add:
     - `GITHUB_TOKEN` — the fine-grained PAT from step 1b (clones the private repo)
     - `GDRIVE_FOLDER_ID` — folder id from step 1
     - your refresh-token secret(s) from step 1: either the single
       `GDRIVE_OAUTH_TOKEN` (Method B), or the three `GDRIVE_REFRESH_TOKEN` +
       `GDRIVE_CLIENT_ID` + `GDRIVE_CLIENT_SECRET` (Method A).

### 3. GitHub repo secrets (for the trigger Action)

On the **private** repo `hardik2015/ace-step-1.5-private` (secrets do **not**
transfer from the old fork) → **Settings → Secrets and variables → Actions →
New repository secret**:

| Secret | Value |
|--------|-------|
| `KAGGLE_USERNAME` | your Kaggle username |
| `KAGGLE_KEY` | the `key` from your Kaggle API token (`kaggle.json`) |

### 4. The Claude routine (`/schedule`)

In Claude Code, run `/schedule` and create a daily routine with this prompt:

> Every day, create one fresh song spec for ACE-Step. Use the `acestep-songwriting`
> skill to write a `title`, a vivid `prompt` (caption: genre, instruments, mood,
> vocals), and complete structured `lyrics` with section tags like `[verse]` /
> `[chorus]`. Vary the genre and mood from day to day and avoid AI-flavored cliché
> lyrics. Write the result to `kaggle/songs.json` (overwrite it) following
> `kaggle/songs.example.json`, and set `"versions": 5` so the kernel renders five
> alternate takes of that one song. Then run:
> `git add kaggle/songs.json && git commit -m "daily lyrics $(date +%F)" && git push origin main`.

- **Schedule:** every day at **7:00 AM IST** → cron `30 1 * * *` (UTC).
- **Connect the routine to the private repo** `hardik2015/ace-step-1.5-private`
  (its `main` is what the kernel clones). `git push origin main` then targets it.
- The routine needs permission to **push to `main`**. If `main` is branch-protected,
  either allow the routine's identity to push, or move the whole flow to a dedicated
  `daily` branch (update the Action's `paths`/`branches` and the notebook's
  `REPO_BRANCH` to match).

---

## The `songs.json` contract

The kernel reads `kaggle/songs.json`. It accepts a single object, a list of
objects, or `{"songs": [ ... ]}`. Required: `prompt` (caption) and `lyrics`.

```json
{
  "title": "Tera Sheher",
  "prompt": "modern Bollywood pop, romantic duet, lush strings, tabla groove, cinematic",
  "lyrics": "[verse]\n...\n[chorus]\n...\n",
  "vocal_language": "unknown",
  "versions": 5
}
```

`versions` (default 1) renders N alternate takes of the *same* song: the kernel
submits it N times on the already-loaded model with random seeds, so each take
differs. Files are saved suffixed `_v01`…`_vNN`, and all are uploaded to Drive.
Because the ~30 min cold start is paid once per run, 5 versions cost only the
extra generation time (≈ `cold start + 5 × per-song`, ~1 h/run — comfortably
inside the weekly quota).

Optional overrides (omit to let the model auto-infer): `bpm`, `keyscale`
(→`key_scale`), `timesignature` (→`time_signature`), `duration` (→`audio_duration`),
`inference_steps`, `guidance_scale`. Aliases are mapped in
`build_notebook.py` → `_build_request_body`.

---

## Picking the best take (automatic scoring)

After generation, the kernel scores every take so you don't have to audition all
five blind. Three signals, all run inside the same Kaggle session:

1. **Broken-take filter (hard gate)** — cheap waveform checks reject duds:
   duration ≥ 20 s, loudness ≥ −35 dBFS (not near-silent), clipping ≤ 2 %,
   silence ≤ 60 %. A take that fails is disqualified (score `−1`) no matter how
   it scores below.
2. **Lyric word-error-rate** — [Whisper](https://github.com/openai/whisper)
   transcribes each surviving take; we compare it to your intended `lyrics`
   (structure tags stripped) and compute WER. Lower = sings your words more clearly.
3. **CLAP style match** — [CLAP](https://huggingface.co/laion/larger_clap_music_and_speech)
   measures text↔audio similarity between your `prompt` (caption) and each take.
   Higher = better matches the requested genre/instruments/mood.

**Final score** = the two signals min-max normalised across the song's passing
takes, then blended `W_LYRIC·(lyric) + W_CLAP·(clap)` (default 0.6 / 0.4).
Normalising makes the two scales comparable; the highest blended score wins. (With
CLAP off, the score is simply `1 − WER`.) The winner is renamed with a **`_BEST`**
suffix (e.g. `..._v03_BEST.mp3`) so it's obvious in Drive, and `manifest.json`
gains per-take `scores` (wer, clap_sim, pass/fail, loudness/clip/silence metrics,
transcript) plus `best_version` / `best_wer` / `best_clap` / `best_path`. All five
takes are still uploaded, so you can A/B by ear.

Config at the top of the notebook cell:

| Setting | Default | Notes |
|---------|---------|-------|
| `SCORE_VERSIONS` | `True` | Turn scoring off entirely. |
| `WHISPER_MODEL` | `"base"` | `"small"`/`"medium"` are more accurate (esp. non-English) but slower. |
| `WHISPER_DEVICE` | `"cpu"` | CPU avoids VRAM contention with the running generator; `"cuda"` is faster if memory allows. |
| `SCORE_CLAP` | `True` | Add CLAP style scoring (falls back to lyric-only if it can't load). |
| `CLAP_MODEL` | `laion/larger_clap_music_and_speech` | Music+vocals CLAP checkpoint (~2 GB, auto-downloaded). |
| `W_LYRIC` / `W_CLAP` | `0.6` / `0.4` | Blend weights (only used when CLAP is active). |

Notes & limits:
- Scoring is **best-effort** — if Whisper or CLAP fails to install/run, the run
  logs a warning and uses whatever signals it has (or uploads unscored); it never
  drops generated audio.
- These rank *lyric clarity* and *style match*, not musical taste. They reliably
  demote mumbled/garbled/off-genre takes and surface a strong shortlist — then
  trust your ear between the top one or two.
- WER is meaningful only for **vocal** tracks; CLAP works for instrumentals too.
- Needs `ffmpeg` (present on Kaggle), `openai-whisper`, and `transformers`
  (auto-installed if missing).

---

## Running it manually / testing

- **Just the trigger:** Actions tab → **Kaggle trigger** → **Run workflow**
  (the `workflow_dispatch` button). Confirms the kernel pushes and runs.
- **End to end:** edit `kaggle/songs.json`, commit, push to `main` → the Action
  fires → watch the kernel at <https://www.kaggle.com/code>. On success, the mp3 +
  `manifest.json` appear in your Drive folder.
- **Regenerate the notebook after editing the builder:**
  `python kaggle/build_notebook.py`.

---

## Operational notes & quotas

- **Cold start every run.** Each Kaggle run is a fresh container: it re-clones the
  repo, runs `uv sync`, and downloads ~13 GB of weights. Expect ~20–40 min/run.
- **Free GPU quota.** Kaggle gives ~30 GPU-hours/week. ~7 daily runs fit, but it's
  the binding constraint — keep songs short or skip days if you approach the cap.
- **Filename.** The kernel notebook is still named `acestep_kaggle_2xT4_weekly.ipynb`
  (historical); it now runs daily.
- **Identical pushes.** The trigger workflow stamps the UTC time into the kernel
  title before pushing so Kaggle always creates a fresh, runnable version.
- **Scoring time.** Whisper + CLAP add ~1–3 min per take on CPU, plus a one-time
  ~2 GB CLAP download per session. For 5 takes that's a few extra minutes per
  run — negligible vs. cold start.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Drive upload fails with `storageQuotaExceeded` | You used a service account against personal Drive. Use `GDRIVE_OAUTH_TOKEN` (this guide), not `GDRIVE_SA_JSON`. |
| `Drive upload needs Kaggle Secrets...` | No usable auth on the kernel. Add `GDRIVE_OAUTH_TOKEN`, or the trio `GDRIVE_REFRESH_TOKEN`+`GDRIVE_CLIENT_ID`+`GDRIVE_CLIENT_SECRET`. |
| `invalid_grant` / token refresh fails | Refresh token revoked/expired (a Testing-status OAuth app expires tokens after 7 days). Re-issue it, or set the consent screen to **Production**. |
| `Set GDRIVE_FOLDER_ID...` | Add the `GDRIVE_FOLDER_ID` Kaggle secret (or set the constant in the notebook). |
| Action runs but no Kaggle run starts | Check `kaggle/kernel-metadata.json` `id` matches your username; verify `KAGGLE_USERNAME`/`KAGGLE_KEY` secrets. |
| Kernel fails at `git clone` (`Authentication failed` / `could not read Username`) | Missing/invalid `GITHUB_TOKEN` Kaggle secret, or the PAT lacks Contents-read on `ace-step-1.5-private`, or it expired. Re-issue the fine-grained PAT (step 1b). |
| Kernel runs but no GPU / out of memory | The kernel's UI Settings lost **T4 ×2**; re-select it (metadata can't set it). |
| Kernel can't download weights | Internet is OFF in the kernel Settings; turn it ON. |
| OAuth setup prints "no refresh_token" | Revoke the app at <https://myaccount.google.com/permissions> and re-run the setup script. |
| Routine commit rejected | `main` is branch-protected; allow the routine to push or switch to a `daily` branch (see setup §4). |

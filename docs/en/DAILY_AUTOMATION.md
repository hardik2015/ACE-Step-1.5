# Daily Auto-Generated Songs (Claude → GitHub → Kaggle → Google Drive)

A hands-off pipeline: every morning at **07:00 IST**, Claude writes fresh lyrics,
commits them to this repo, a **Kaggle 2×T4** kernel generates the song, and the
finished audio lands in a **Google Drive folder**.

```
  Daily 07:00 IST   Claude Code Routine  (/schedule, connected to the private repo)
    • writes kaggle/songs.json   • git commit + push → main
        │  (push touches kaggle/songs.json)
        ▼
  GitHub Action  kaggle-trigger.yml   (fire-and-forget, ~2 min)
    1. build_notebook.py → notebook with __PLACEHOLDER__ secrets
    2. replace placeholders with real GitHub repo secret values (no echo)
    3. kaggle kernels push --accelerator NvidiaTeslaT4  → creates + runs on T4×2
        │  (then exits — the run continues on Kaggle)
        ▼
  Kaggle kernel  (2×T4, internet ON)   — finishes on its own (~20–40 min)
    • git clone PRIVATE repo (via injected GITHUB_TOKEN) → reads songs.json
    • generates 5 takes → score (filter + WER + CLAP) → mark _BEST
    • uploads mp3s + manifest.json → Google Drive folder
        │
        ▼
  Your Google Drive folder  ← finished songs appear here daily
        ⋮
  GitHub Action  kaggle-cleanup.yml   (scheduled ~11:00 IST, or manual)
    • deletes the kernel → purges the injected secrets
      (skips if a run is still in progress, so it can't kill it)
```

**Why inject-and-delete, not Kaggle's secret store?** Two Kaggle-API limits shaped
this: `kaggle kernels push` (1) **unlinks the kernel's attached Secrets every push**
(so a pushed run can't read them) and (2) can't request T4×2 by itself. So instead
of relying on Kaggle secrets, the Action **bakes the secret values into the notebook**
just before pushing (placeholders → real values from GitHub repo secrets), pushes
with `--accelerator NvidiaTeslaT4` (Kaggle's T4 tier = 2 GPUs), and exits — the run
finishes on Kaggle on its own. A **separate cleanup workflow** then **deletes the
whole kernel** a few hours later so the values don't persist. The next day's push
recreates it (confirmed: push recreates a deleted kernel). All secrets live only in
**GitHub repo secrets** (encrypted) — never in Kaggle's stored notebook long-term.

---

## What's in the repo

| File | Role |
|------|------|
| `kaggle/build_notebook.py` | Generates the single-cell kernel notebook. Defaults are now `INPUT_MODE="file"`, `OUTPUT_MODE="gdrive"`. |
| `kaggle/acestep_kaggle_2xT4_weekly.ipynb` | The kernel notebook — **committed** to the repo. It's generated from `build_notebook.py`; if you edit the builder, run it and commit the regenerated `.ipynb`. |
| `kaggle/songs.json` | **The daily artifact.** The routine overwrites this; the kernel reads it. |
| `kaggle/songs.example.json` | Schema documentation (not read by the kernel). |
| `kaggle/kernel-metadata.json` | Required by `kaggle kernels push`. Put your Kaggle username in `id`. |
| `.github/workflows/kaggle-trigger.yml` | The daily trigger: on push to `kaggle/songs.json` it injects secrets and pushes+runs on T4×2 (fire-and-forget). |
| `.github/workflows/kaggle-cleanup.yml` | Scheduled (~11:00 IST) + manual: deletes the kernel to purge the injected secrets once the run has finished. |
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
5. You now have all three values. Set them as **three GitHub repo secrets**
   (step 2): `GDRIVE_REFRESH_TOKEN`, `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`.

**Method B — local helper script (one bundled secret):**
1. Create the client as type **Desktop app** → download the JSON.
2. Locally:
   ```bash
   pip install google-auth-oauthlib google-auth
   python scripts/gdrive_oauth_setup.py path/to/client_secret.json
   ```
   A browser opens; approve. The script prints an **authorized-user JSON** (it
   contains the refresh token) — set it as the single GitHub repo secret `GDRIVE_OAUTH_TOKEN`.

**Then (both methods):** in Google Drive, create/open the destination folder. Its
**folder id** is the part after `/folders/` in the URL → GitHub repo secret `GDRIVE_FOLDER_ID`.

### 1b. GitHub token (the repo is PRIVATE)

The pipeline lives in the **private** repo `hardik2015/ace-step-1.5-private`, so the
kernel needs a token to clone it.

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. **Resource owner** = your account; **Repository access** = *Only select
   repositories* → `ace-step-1.5-private`; **Permissions → Repository → Contents
   = Read-only**. Generate and copy the token (`github_pat_...`).
3. Add it as the GitHub repo secret **`MAIN_GITHUB_TOKEN`** in step 2. (It can't be
   named `GITHUB_TOKEN` — that name is reserved by GitHub Actions.)

### 2. GitHub repo secrets (this is where ALL secrets live)

The Action bakes these into the notebook at push time, then deletes the kernel
afterward — so nothing is stored on Kaggle. On the **private** repo →
**Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|--------|-------|
| `KAGGLE_API_TOKEN` | Kaggle → Settings → API → API Tokens (Recommended) → create, copy |
| `MAIN_GITHUB_TOKEN` | the fine-grained PAT from step 1b (clones the private repo) |
| `GDRIVE_FOLDER_ID` | your Drive folder id |
| `GDRIVE_OAUTH_TOKEN` **or** the trio `GDRIVE_REFRESH_TOKEN` + `GDRIVE_CLIENT_ID` + `GDRIVE_CLIENT_SECRET` | the Drive auth from step 1 |
| `GDRIVE_FOLDER_ID_KIDS` *(optional)* | Drive folder for kids songs (`category: kids`); falls back to `GDRIVE_FOLDER_ID` if unset |
| `KIDS_APP_URL` + `KIDS_API_KEY` *(optional)* | kids-songs-web deployment URL + its `API_SECRET_KEY`. When both set, a kids run posts every take's Drive file id to `<KIDS_APP_URL>/api/song-file` (matched by song title) so the combine page can pick a version — no manual paste. Unset → skipped. |

No Kaggle-side secrets, no accelerator/internet toggles, no scheduler — the push
sets the accelerator (`--accelerator NvidiaTeslaT4`) and internet
(`enable_internet: true`), and injects the secrets.

### 3. (Nothing else) — two workflows handle it

- **`kaggle-trigger.yml`** runs on every commit to `kaggle/songs.json` (plus a
  manual **Run workflow** button). It regenerates the notebook, injects the secrets
  above, pushes on T4×2, and exits (~2 min). The Kaggle run finishes on its own.
- **`kaggle-cleanup.yml`** runs on a daily schedule (~11:00 IST, plus manual). It
  **deletes the kernel** to purge the injected secrets — skipping if a run is still
  in progress so it can't kill it. The next day's trigger recreates the kernel.

You don't configure anything on Kaggle.

> Security note: between the run and the cleanup, the secret values sit inside the
> **private** Kaggle kernel; the cleanup deletes the whole kernel to remove them.
> Keep the kernel private and the credentials minimally scoped. If you want the
> exposure window shorter, lower the cleanup schedule (but leave enough time for
> the ~20–40 min run to finish).

### 4. The Claude routine (`/schedule`)

In Claude Code, run `/schedule` and create a daily routine with this prompt:

> Every day, create one fresh song spec for ACE-Step. Use the `acestep-songwriting`
> skill to write a `title`, a vivid `prompt` (caption: genre, instruments, mood,
> vocals), and complete structured `lyrics` with section tags like `[verse]` /
> `[chorus]`. Vary the genre and mood from day to day and avoid AI-flavored cliché
> lyrics. Write the result to `kaggle/kidssong.json` (overwrite it) following
> `kaggle/songs.example.json`, set `"category": "kids"` (so the song-repo pipeline
> takes the kids path and fires the Vecteezy video search), and set `"versions": 5`
> so the kernel renders five takes you can pick between on the combine page. Then run:
> `git add kaggle/kidssong.json && git commit -m "daily kids song $(date +%F)" && git push origin main`.

- **Schedule:** routine at **07:00 IST** (cron `30 1 * * *` UTC). Its commit fires
  the GitHub Action immediately (event-driven) — no Kaggle scheduling needed.
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
  "prompt": "modern Bollywood pop with dholak groove, soft bansuri and lush strings, romantic and cinematic, polished studio production, tender male-female duet vocals, warm lush timbre",
  "lyrics": "[Intro]\n...\n[Verse]\n...\n[Chorus]\n...\n[Outro]\n",
  "vocal_language": "hindi",
  "bpm": 95,
  "key_scale": "C Major",
  "time_signature": "4/4",
  "audio_duration": 200,
  "versions": 1
}
```

> **The caption (`prompt`) is ~70% of output quality.** Make it multi-dimensional
> (genre · 3-4 instruments · mood · production · vocal style · timbre), not a
> generic line — a thin caption is what makes songs sound generic/over-lyrical.
> The daily routine targets a **Bollywood/Hindi-primary hybrid** (≈3 Indian : 1
> global-English) for distribution reach, favouring modern fusion + mid-tempo and
> capping `audio_duration` ≈180-210s (ACE-Step vocals are coarse on nuance-heavy
> classical, and genre can drift past ~2 min).

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

- **Run it now:** GitHub → **Actions → "Kaggle daily run" → Run workflow**
  (`workflow_dispatch`). It injects secrets, pushes on T4×2, waits, then deletes the
  kernel. Watch the kernel live at
  <https://www.kaggle.com/code/allinone2015/acestep-daily> while it runs; the mp3s +
  `manifest.json` land in your Drive folder.
- **End to end:** edit `kaggle/songs.json`, commit, push to `main` → the Action
  fires automatically.
- **Read the Action log** to follow it: `replaced N placeholder(s)` → `successfully
  pushed` → the status loop → kernel deleted.

---

## Operational notes & quotas

- **Cold start every run.** Each Kaggle run is a fresh container: it re-clones the
  repo, runs `uv sync`, and downloads ~13 GB of weights. Expect ~20–40 min/run.
- **Free GPU quota.** Kaggle gives ~30 GPU-hours/week. ~7 daily runs fit, but it's
  the binding constraint — keep songs short or skip days if you approach the cap.
- **Filename.** The kernel notebook is still named `acestep_kaggle_2xT4_weekly.ipynb`
  (historical); it now runs daily.
- **Fresh run guaranteed.** Because the Action deletes the kernel after each run,
  the next push always *creates* it (version 1) and runs — no stale/no-op pushes.
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
| Action runs but no Kaggle run starts | Check `kaggle/kernel-metadata.json` `id` matches your username; verify the `KAGGLE_API_TOKEN` secret. |
| `401 - Unauthorized` on `kaggle kernels push` | `KAGGLE_API_TOKEN` is missing/invalid/expired. Re-create it under Kaggle Settings → API Tokens and update the repo secret. |
| Kernel fails at `git clone` (`Authentication failed` / `could not read Username`) | `MAIN_GITHUB_TOKEN` repo secret missing/invalid, the PAT lacks **Contents: Read** on `ace-step-1.5-private`, or it expired. The Action's `replaced N placeholder(s)` line shows how many secrets were injected — if `N` is low/0, the repo secrets are missing. |
| Kernel disappears from Kaggle after each run | Expected — the Action deletes it to purge the injected secrets. The next commit/run recreates it (version 1 each time). |
| Kernel ran on a single **P100** (no `cuda:1`) and failed | The push didn't request T4×2. Ensure the Action runs `kaggle kernels push … --accelerator NvidiaTeslaT4` (Kaggle's T4 tier = 2 GPUs). |
| Kernel can't download weights | Internet is OFF in the kernel Settings; turn it ON. |
| OAuth setup prints "no refresh_token" | Revoke the app at <https://myaccount.google.com/permissions> and re-run the setup script. |
| Routine commit rejected | `main` is branch-protected; allow the routine to push or switch to a `daily` branch (see setup §4). |

# Daily Auto-Generated Songs (Claude → GitHub → Kaggle → Google Drive)

A hands-off pipeline: every morning at **07:00 IST**, Claude writes fresh lyrics,
commits them to this repo, a **Kaggle 2×T4** kernel generates the song, and the
finished audio lands in a **Google Drive folder**.

```
  Daily 07:00 IST   Claude Code Routine  (/schedule, connected to the private repo)
    • writes kaggle/songs.json   • git commit + push → main
        │  (just updates the repo)
        ▼
  Daily 07:30 IST   Kaggle built-in SCHEDULER re-runs the saved kernel version
    • uses the UI-configured T4×2 + Internet + attached Secrets
    • git clone PRIVATE repo (main, via GITHUB_TOKEN)  → gets the new songs.json
    • reads kaggle/songs.json   (INPUT_MODE="file")
    • acestep-api generates 5 takes → score (filter + WER + CLAP) → mark _BEST
    • uploads mp3s + manifest.json → Google Drive folder
    • Secrets (attached in the UI): GITHUB_TOKEN, GDRIVE_FOLDER_ID,
      GDRIVE_OAUTH_TOKEN (or the refresh trio)
        │
        ▼
  Your Google Drive folder  ← finished songs appear here daily
```

**Why Kaggle's scheduler, not a GitHub-Action push?** Two Kaggle-API limits make
`kaggle kernels push` unusable as the daily trigger: (1) it can't request **T4×2**
(only a single P100, where `cuda:1` doesn't exist and the 2-GPU notebook fails);
and decisively (2) **it unlinks the kernel's attached Secrets on every push**, so a
pushed run can't read `GITHUB_TOKEN` / `GDRIVE_*` and the clone + Drive upload fail.
Kaggle's UI runs and its built-in **scheduler** keep the configured accelerator,
internet, and secrets — so the scheduler re-runs the saved, fully-configured
version daily. The `/schedule` routine just commits `songs.json`; the kernel clones
the latest repo when the scheduler fires.

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

### 2. Kaggle kernel: configure, run once, and schedule (all in the UI)

The kernel `allinone2015/acestep-daily` already has the code. From here everything
is done in the **Kaggle UI** — do **not** `kaggle kernels push` again (each push
unlinks your secrets). Open the notebook editor and:

1. **Settings:** Accelerator = **GPU T4 ×2**, Internet = **ON**.
2. **Add-ons → Secrets:** attach (toggle ON) — labels must match **exactly**:
   - `GITHUB_TOKEN` — fine-grained PAT from step 1b (Contents: Read on the repo)
   - `GDRIVE_FOLDER_ID` — folder id from step 1
   - `GDRIVE_OAUTH_TOKEN`, **or** the trio `GDRIVE_REFRESH_TOKEN` + `GDRIVE_CLIENT_ID` + `GDRIVE_CLIENT_SECRET`

   Press **Done** so they're attached to this notebook.
3. **Save Version → Save & Run All (Commit)** once. Watch the Logs — the clone
   should succeed and a song should land in Drive. This proves the config.
4. **Schedule it (the daily trigger):** **Save Version → Schedule a notebook run →
   Daily**, time ≈ **07:30 IST** (≈30 min after the routine commits). Scheduled
   runs reuse this saved version with its T4×2 + internet + attached secrets, and
   clone the latest repo each morning to pick up that day's `songs.json`.

### 3. The GitHub Action is manual-only (not the daily trigger)

`kaggle-trigger.yml` no longer runs on commits — it's `workflow_dispatch`-only, for
re-deploying notebook **code** changes. Because a push **unlinks secrets**, after
running it you must re-open **Add-ons → Secrets** in the UI and press **Done** to
re-attach them (then the next scheduled run works again). For normal daily
operation you never touch it. It needs the `KAGGLE_API_TOKEN` repo secret only if
you use it.

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

- **Schedule:** routine at **07:00 IST** (cron `30 1 * * *` UTC); set Kaggle's
  notebook schedule ~30 min later (≈07:30 IST) so the commit lands first.
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

- **Run it now:** in the Kaggle editor → **Save Version → Save & Run All (Commit)**.
  Uses the UI-attached secrets + T4×2. Watch the Logs; the mp3s + `manifest.json`
  appear in your Drive folder on success.
- **Daily:** Kaggle's scheduler re-runs that saved version (≈07:30 IST); the routine
  commits the day's `songs.json` first (≈07:00 IST), which the run clones.
- **Re-deploy notebook code changes:** regenerate (`python kaggle/build_notebook.py`),
  then run the manual `Kaggle deploy` Action (or push locally) — and afterward
  **re-attach the secrets** in Add-ons → Secrets → Done (the push unlinks them).

---

## Operational notes & quotas

- **Cold start every run.** Each Kaggle run is a fresh container: it re-clones the
  repo, runs `uv sync`, and downloads ~13 GB of weights. Expect ~20–40 min/run.
- **Free GPU quota.** Kaggle gives ~30 GPU-hours/week. ~7 daily runs fit, but it's
  the binding constraint — keep songs short or skip days if you approach the cap.
- **Filename.** The kernel notebook is still named `acestep_kaggle_2xT4_weekly.ipynb`
  (historical); it now runs daily.
- **Identical pushes.** The trigger workflow appends a `# run-stamp:` comment to
  the notebook before pushing, so Kaggle always sees fresh content and creates a
  new, runnable version (the title is left alone so it keeps matching the slug).
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
| Kernel fails at `git clone` (`Authentication failed` / `could not read Username`) | `GITHUB_TOKEN` was empty/invalid: secret not attached to this notebook, **detached by a `kaggle kernels push`** (re-attach in Add-ons → Secrets → Done), Internet OFF (`get_secret` needs it), label mismatch, or the PAT lacks Contents-read / expired. |
| Worked, then broke right after a code re-deploy | A `kaggle kernels push` **unlinks secrets**. Re-open Add-ons → Secrets → press **Done**, then let the scheduler run again. |
| Kernel ran on a single **P100** (no `cuda:1`) and failed | The push didn't request T4×2. Ensure the Action runs `kaggle kernels push … --accelerator NvidiaTeslaT4` (Kaggle's T4 tier = 2 GPUs). |
| Kernel can't download weights | Internet is OFF in the kernel Settings; turn it ON. |
| OAuth setup prints "no refresh_token" | Revoke the app at <https://myaccount.google.com/permissions> and re-run the setup script. |
| Routine commit rejected | `main` is branch-protected; allow the routine to push or switch to a `daily` branch (see setup §4). |

"""Generate the ACE-Step single-cell Kaggle notebook (.ipynb).

The output notebook has ONE code cell that:
  - applies the stability env from the working ace-step.ipynb
  - clones the user's fork to /kaggle/tmp, runs `uv sync`
  - launches the project's `acestep-api` FastAPI server (background)
  - submits each song via POST /release_task using the `acestep-v15-sft`
    DiT model, then polls POST /query_result until the job leaves
    status 0 (queued/running) -> 1 (succeeded) | 2 (failed)
  - audio file paths returned by the server are copied (or fetched via
    GET /v1/audio when the local path is unreachable) into LOCAL_OUT,
    or uploaded to a Google Drive folder when OUTPUT_MODE="gdrive"
    (personal My Drive via OAuth token, or a Shared Drive via service account)

This is the cleanest possible approach: the FastAPI server already speaks
JSON in/out (GenerateMusicRequest), so we drop the 78-field positional
Gradio contract entirely.

Run:  python kaggle/build_notebook.py
Output: kaggle/acestep_kaggle_2xT4_weekly.ipynb
"""
import json
import os


# ---------------------------------------------------------------------------
# The notebook cell — orchestration only.
# ---------------------------------------------------------------------------
NOTEBOOK_CELL_TEMPLATE = r'''# =============================================================================
# ACE-Step 2x T4 - single-cell headless generator (REST API driven)
#
# Launches the project's FastAPI server (`acestep-api`, registered in
# pyproject.toml -> acestep.api_server:main) and submits jobs via HTTP.
# Uses the `acestep-v15-sft` DiT model. Per-song fields override defaults.
#
# Prereqs:
#   - Kaggle: Settings -> Accelerator = GPU T4 x2, Internet = ON
#   - Drive upload (OUTPUT_MODE="gdrive"): add the destination folder id as the
#     constant GDRIVE_FOLDER_ID below or the Kaggle secret GDRIVE_FOLDER_ID, plus
#     ONE auth secret (auto-detected):
#       GDRIVE_OAUTH_TOKEN -> personal "My Drive": an OAuth authorized-user JSON
#                             (client_id/client_secret/refresh_token) minted once
#                             by scripts/gdrive_oauth_setup.py. Personal Drive
#                             folders MUST use this (service accounts can't write
#                             to My Drive -- 0-byte quota -> storageQuotaExceeded).
#                             Alternative (e.g. from the Google OAuth Playground):
#                             supply GDRIVE_REFRESH_TOKEN + GDRIVE_CLIENT_ID +
#                             GDRIVE_CLIENT_SECRET as three separate secrets.
#       GDRIVE_SA_JSON     -> Shared Drive: a service-account JSON. The target
#                             folder must live in a Google Workspace Shared Drive.
# =============================================================================

# ----- CONFIG (edit these) ---------------------------------------------------
REPO_URL    = "https://github.com/hardik2015/ACE-Step-1.5.git"
REPO_BRANCH = "main"
WORK_DIR    = "/kaggle/tmp/ACE-Step-1.5"

OUTPUT_MODE       = "gdrive"                               # "local" | "gdrive"
LOCAL_OUT         = "/kaggle/working/acestep_output"
GDRIVE_FOLDER_ID  = ""                                     # Drive folder id (or Kaggle secret)

INPUT_MODE        = "file"                                 # "inline" | "file"
INPUT_JSON_PATH   = f"{WORK_DIR}/kaggle/songs.json"        # committed daily by the Claude routine

SONG_JSON_INLINE = r"""
{
  "title":  "Tera Sheher",
  "prompt": "modern Bollywood pop, romantic duet male and female vocals, lush strings, tabla and dholak groove, soft flute, cinematic and emotional, danceable mid-tempo",
  "lyrics": "[verse]\nSham dhale teri yaadein chali aayi\nTere bina yeh duniya soti nahi\nEvery street feels like a song we used to know\n[chorus]\nTera sheher teri galiyan\nTere naam ki yeh raahein\nHold me close tonight\nRoshni ban ke aaja zindagi mein\n",
  "vocal_language": "unknown"
}
"""

# Defaults merged UNDER each song. Field names match GenerateMusicRequest
# (acestep/api/http/release_task_models.py). Per-song JSON overrides win.
GEN_DEFAULTS = {
    "model":              "acestep-v15-sft",   # SFT DiT (registered in model_download.py)
    "thinking":           True,                # use 5Hz LM to generate audio codes
    "use_cot_caption":    False,               # keep YOUR caption verbatim
    "use_cot_language":   True,
    "use_format":         False,
    "inference_steps":    50,
    "guidance_scale":     7.0,
    "use_random_seed":    True,
    "audio_format":       "mp3",
    "batch_size":         1,
    "lm_backend":         "pt",                # matches STABILITY_ENV layout below
    "lm_cfg_scale":       2.0,
    # No "audio_duration" => the LM picks (consistent codes/latents).
}

# Multi-GPU layout (matches the working ace-step.ipynb).
# ACESTEP_CONFIG_PATH selects the primary DiT model loaded at startup; we point
# it at the SFT checkpoint so requests routed to "acestep-v15-sft" hit it.
#
# MAX_CUDA_VRAM: the GPU "tier" gate (acestep/gpu_config.py) sees ONE T4's
# 14.6GB -> tier5, which forbids the 4B LM and falls back to acestep-5Hz-lm-1.7B
# -- a model that is NOT in SUBMODEL_REGISTRY and not in the unified weights
# bundle, so init hard-fails ("5Hz LM model not found"). The 4B IS bundled and
# the LM gets its OWN dedicated T4 (cuda:1), so it fits fine; the gate is just
# wrongly assuming DiT+LM share one GPU. Reporting 24GB promotes the gate to
# tier6b, which permits the 4B we already have on disk. The DiT still loads
# float32/persistent on cuda:0, BUT tier6b also flips offload_to_cpu_default to
# False -- the auto-offload gate (startup_model_init.py) keys off this SAME
# simulated VRAM (auto_offload = vram < 20), so 24 silently disables the
# VAE/text-encoder CPU offload the real 14.6GB card needs. We pin
# ACESTEP_OFFLOAD_TO_CPU=true to restore the tier5 offload (the explicit env
# override wins over the tier default); without it cuda:0 OOMs once batch size
# or duration grows.
STABILITY_ENV = {
    "ACESTEP_CONFIG_PATH":         "acestep-v15-sft",
    "ACESTEP_DTYPE":               "float32",
    "MAX_CUDA_VRAM":               "24",       # unlock the bundled 4B LM (see note above)
    "ACESTEP_OFFLOAD_TO_CPU":      "true",     # restore VAE/TextEnc offload the 24GB lie disables
    "ACESTEP_LM_DEVICE":           "cuda:1",
    "ACESTEP_LM_BACKEND":          "pt",
    "NANOVLLM_DISABLE_CUDA_GRAPH": "1",
    "ACE_DISABLE_CUDA_GRAPHS":     "1",
    "VLLM_DISABLE_CUDA_GRAPH":     "1",
    "VLLM_ENFORCE_EAGER":          "1",
    "TORCHDYNAMO_DISABLE":         "1",
    "TORCHAO_FORCE_FP32":          "1",
    "PYTORCH_CUDA_ALLOC_CONF":     "expandable_segments:True",
    "MPLBACKEND":                  "agg",
    "CUDA_LAUNCH_BLOCKING":        "1",
    "ACESTEP_GENERATION_TIMEOUT":  "1800",
}

API_HOST = "127.0.0.1"
API_PORT = 8001
API_BASE = f"http://{API_HOST}:{API_PORT}"
API_LOG  = "/kaggle/working/acestep_api.log"
API_READY_TIMEOUT_S = 60 * 30           # first run downloads ~13GB weights

# Poll cadence for /query_result (status 0 = queued/running, 1 = succeeded, 2 = failed)
JOB_POLL_INTERVAL_S = 5
JOB_TIMEOUT_S       = 60 * 60           # per-song wall clock budget

# Version scoring (runs after generation). A broken-take filter (clipping /
# silence / loudness / duration) gates out duds, then two signals rank the
# survivors: (1) Whisper lyric word-error-rate vs the intended lyrics, and
# (2) CLAP text<->audio similarity vs the caption (how well the take matches the
# requested STYLE). The two are min-max normalised per song and blended by the
# weights below. The winner is renamed with a _BEST suffix. Best-effort: any
# failure here just skips ranking, it never drops a generated file.
SCORE_VERSIONS = True
WHISPER_MODEL  = "base"   # "base" fast | "small"/"medium" more accurate (multilingual)
WHISPER_DEVICE = "cpu"    # CPU avoids VRAM contention with the still-running API server
SCORE_CLAP     = True     # add CLAP style-match scoring
CLAP_MODEL     = "laion/larger_clap_music_and_speech"   # music+vocals CLAP checkpoint
CLAP_DEVICE    = "cpu"
W_LYRIC        = 0.6      # blend weights for the final score (lyric clarity ...
W_CLAP         = 0.4      # ... vs. style match). Only used when CLAP is active.

# ----- IMPLEMENTATION --------------------------------------------------------
import os, sys, json, time, shutil, subprocess, pathlib, datetime
import http.client
import urllib.request
import urllib.parse
import urllib.error

# Transient HTTP failures we tolerate while polling /query_result. The server
# runs generation in a thread pool, but the worker thread holds the GIL almost
# solidly while loading weights / driving the LM token loop, which starves
# uvicorn's event loop -> polls time out even though the job is healthy.
# urllib.error.URLError and socket timeouts are OSError subclasses; http.client
# raises HTTPException (e.g. RemoteDisconnected/BadStatusLine) on a dropped conn.
TRANSIENT_HTTP_ERRORS = (OSError, http.client.HTTPException)


def sh(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=os.environ.copy())
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {cmd}")


# 1) Apply env (matches the working ace-step.ipynb).
for k, v in STABILITY_ENV.items():
    os.environ[k] = v
print("env:", " ".join(f"{k}={v}" for k, v in STABILITY_ENV.items()))

# 2) Show GPUs.
try:
    print(subprocess.check_output(
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
        shell=True, text=True))
except Exception as exc:
    print(f"(nvidia-smi unavailable: {exc})")

# 3) Install uv if missing.
if not shutil.which("uv"):
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh", check=False)
os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")

# 4) Clone / update repo into /kaggle/tmp.
os.makedirs(os.path.dirname(WORK_DIR), exist_ok=True)
if not os.path.isdir(os.path.join(WORK_DIR, ".git")):
    sh(f"git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} {WORK_DIR}")
else:
    sh(f"git -C {WORK_DIR} fetch --depth 1 origin {REPO_BRANCH}", check=False)
    sh(f"git -C {WORK_DIR} checkout {REPO_BRANCH}", check=False)
    sh(f"git -C {WORK_DIR} reset --hard origin/{REPO_BRANCH}", check=False)

# 5) Build the project's .venv with pinned torch.
sh("uv sync", cwd=WORK_DIR, check=False)

# 6) Launch acestep-api in the background using the project .venv.
def _api_alive():
    try:
        with urllib.request.urlopen(f"{API_BASE}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

if not _api_alive():
    log_f = open(API_LOG, "w")
    cmd = ["uv", "run", "acestep-api",
           "--host", API_HOST,
           "--port", str(API_PORT),
           "--lm-model-path", "acestep-5Hz-lm-4B"]
    print("launching:", " ".join(cmd), f"(logs -> {API_LOG})")
    proc = subprocess.Popen(cmd, cwd=WORK_DIR, env=os.environ.copy(),
                            stdout=log_f, stderr=subprocess.STDOUT)
    print(f"acestep-api pid={proc.pid}, waiting up to {API_READY_TIMEOUT_S}s for /health ...")
    deadline = time.time() + API_READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            print("acestep-api exited early. Tail of log:")
            with open(API_LOG) as f:
                print("".join(f.readlines()[-80:]))
            raise RuntimeError("acestep-api failed to start")
        if _api_alive():
            break
        time.sleep(5)
    else:
        raise TimeoutError("acestep-api not ready in time")
print(f"acestep-api ready on {API_BASE}")


# 7) HTTP helpers (no extra dependencies; uses urllib).
def _api_post(path, body, timeout=60):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _print_api_log_tail(n=160):
    """Dump the tail of the acestep-api log so the REAL failure traceback shows.

    /query_result only echoes status/progress/stage from the progress cache and
    drops the server-side error string, so on failure the log file is the only
    place the actual exception is recorded.
    """
    try:
        with open(API_LOG, "r", errors="replace") as f:
            lines = f.readlines()
        shown = lines[-n:]
        print(f"---- {API_LOG} (last {len(shown)} of {len(lines)} lines) ----")
        print("".join(shown).rstrip())
        print("---- end of acestep-api log ----")
    except Exception as exc:
        print(f"(could not read {API_LOG}: {exc})")


# 8) Resolve song input.
if INPUT_MODE == "file":
    with open(INPUT_JSON_PATH) as f:
        songs_data = json.load(f)
else:
    songs_data = json.loads(SONG_JSON_INLINE)
if isinstance(songs_data, dict):
    songs = songs_data["songs"] if isinstance(songs_data.get("songs"), list) else [songs_data]
elif isinstance(songs_data, list):
    songs = songs_data
else:
    raise ValueError("Song input must be a dict, a list, or {'songs':[...]}")
print(f"songs: {len(songs)}")


def _build_request_body(song):
    """Merge GEN_DEFAULTS + per-song overrides into a GenerateMusicRequest body.

    Maps common aliases (caption/captions -> prompt, duration -> audio_duration,
    keyscale -> key_scale, timesignature -> time_signature) for ergonomics.
    """
    aliases = {
        "caption":        "prompt",
        "captions":       "prompt",
        "keyscale":       "key_scale",
        "timesignature":  "time_signature",
        "duration":       "audio_duration",
    }
    body = dict(GEN_DEFAULTS)
    for k, v in song.items():
        if k in ("title", "versions"):   # local-only keys, not GenerateMusicRequest fields
            continue
        body[aliases.get(k, k)] = v
    return body


def _wait_for_job(task_id, deadline):
    """Poll /query_result until the job leaves status 0. Returns the parsed
    result list (a list of dicts with 'file', 'metas', etc.)."""
    last_progress = ""
    transient = 0
    while time.time() < deadline:
        try:
            # Generous read timeout: the event loop can stall for a while during
            # weight loading / LM decoding. A timeout here is NOT job failure.
            resp = _api_post("/query_result", {"task_id_list": [task_id]}, timeout=120)
        except TRANSIENT_HTTP_ERRORS as exc:
            transient += 1
            print(f"    .. poll retry #{transient} ({type(exc).__name__}: {exc}); "
                  f"server busy, {int(deadline - time.time())}s budget left")
            time.sleep(JOB_POLL_INTERVAL_S)
            continue
        transient = 0
        items = (resp or {}).get("data") or []
        if not items:
            time.sleep(JOB_POLL_INTERVAL_S)
            continue
        item = items[0]
        status = item.get("status", 0)
        progress = item.get("progress_text") or ""
        if progress and progress != last_progress:
            print(f"    .. {progress}")
            last_progress = progress
        if status == 1:
            try:
                return json.loads(item.get("result") or "[]")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"could not parse result JSON: {exc}") from exc
        if status == 2:
            # The cache-derived result payload drops the server error string,
            # so surface the last log line + the full server log tail.
            progress_text = item.get("progress_text") or ""
            inner_err = ""
            try:
                rd = json.loads(item.get("result") or "[]")
                if isinstance(rd, list) and rd and isinstance(rd[0], dict):
                    inner_err = rd[0].get("error") or ""
            except Exception:
                pass
            if progress_text:
                print(f"    !! last log: {progress_text}")
            if inner_err:
                print(f"    !! error: {inner_err}")
            _print_api_log_tail()
            raise RuntimeError(
                f"job {task_id} failed (real cause in the acestep-api log above)"
            )
        time.sleep(JOB_POLL_INTERVAL_S)
    raise TimeoutError(f"job {task_id} did not finish within {JOB_TIMEOUT_S}s")


# 9) Run generation. Each song may request N "versions" (alternate takes) via a
# `versions` field. We submit the SAME song N times on the already-loaded model,
# keeping batch_size=1 so a 14.6GB T4 never OOMs (true batching would). Random
# seeds (use_random_seed) make every take differ. The ~30min cold start is paid
# once for the whole run regardless of N, so extra versions cost only gen time.
os.makedirs(LOCAL_OUT, exist_ok=True)
manifest = []
for i, song in enumerate(songs, 1):
    title = (song.get("title") or song.get("prompt") or song.get("caption") or f"song{i}")[:80]
    n_versions = max(1, int(song.get("versions", 1) or 1))
    body = _build_request_body(song)            # "versions" is stripped inside
    print(f"\n[gen {i}/{len(songs)}] {title!r}  x{n_versions} version(s)")

    versions = []
    for v in range(1, n_versions + 1):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"  -- version {v}/{n_versions}")
        submit = _api_post("/release_task", body)
        submit_data = (submit or {}).get("data") or {}
        task_id = submit_data.get("task_id")
        if not task_id:
            raise RuntimeError(f"release_task did not return task_id: {submit}")
        print(f"    task_id={task_id} queue_position={submit_data.get('queue_position')}")

        deadline = time.time() + JOB_TIMEOUT_S
        results = _wait_for_job(task_id, deadline)

        saved = []
        for ai, item in enumerate(results, 1):
            file_ref = item.get("file") or ""
            if not file_ref:
                continue
            # The server returns `file` as a ready-made URL, NOT a raw path:
            #   "/v1/audio?path=<urlencoded filesystem path>"
            # Recover the real path so we can copy it directly (the API server shares
            # this container). Only if that fails do we fetch the URL as-is -- we must
            # NOT pass file_ref back through /v1/audio?path=, because re-wrapping
            # double-encodes the path and the server rejects it with 403.
            parsed = urllib.parse.urlparse(file_ref)
            local_path = urllib.parse.parse_qs(parsed.query).get("path", [""])[0]
            ext = pathlib.Path(local_path or file_ref).suffix or f".{GEN_DEFAULTS['audio_format']}"
            dst = os.path.join(LOCAL_OUT, f"{stamp}_{i:03d}_v{v:02d}_{ai}{ext}")
            if local_path and os.path.exists(local_path):
                shutil.copy2(local_path, dst)            # same container: just copy
            else:
                url = file_ref if parsed.scheme else f"{API_BASE}{file_ref}"
                with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as out:
                    shutil.copyfileobj(r, out)
            print(f"   -> {dst} ({os.path.getsize(dst)//1024} KB)")
            saved.append(dst)
        versions.append({"version": v, "task_id": task_id,
                         "paths": saved, "success": bool(saved)})

    manifest.append({
        "index": i,
        "title": title,
        "versions": versions,
        "success": any(x["success"] for x in versions),
    })

# 9.5) Score the takes: broken-take filter (hard gate) + Whisper lyric-WER.
import re


def _norm_words(s):
    """Lowercase, drop [structure tags], strip punctuation -> word list."""
    s = re.sub(r"\[[^\]]*\]", " ", s or "")          # remove [verse]/[chorus]/hints
    s = re.sub(r"[^\w\s]", " ", s.lower(), flags=re.UNICODE)
    return [w for w in s.split() if w]


def _wer(ref, hyp):
    """Word error rate = Levenshtein(ref, hyp) / len(ref)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def _audio_metrics(wav, sr):
    """Cheap health metrics on a mono float waveform in [-1, 1]."""
    import numpy as np
    if wav.size == 0:
        return {"duration_s": 0.0, "rms_dbfs": -120.0, "peak": 0.0,
                "clip_ratio": 1.0, "silence_ratio": 1.0}
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-12)
    fl = max(1, int(0.05 * sr))                       # 50 ms frames
    nf = wav.size // fl
    if nf:
        fr = wav[:nf * fl].reshape(nf, fl)
        fr_rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
        silence_ratio = float(np.mean(fr_rms < 10 ** (-40 / 20)))   # < -40 dBFS
    else:
        silence_ratio = 1.0
    return {
        "duration_s":    round(wav.size / sr, 2),
        "rms_dbfs":      round(20 * float(np.log10(rms)), 1),
        "peak":          round(float(np.max(np.abs(wav))), 4),
        "clip_ratio":    round(float(np.mean(np.abs(wav) > 0.99)), 4),
        "silence_ratio": round(silence_ratio, 3),
    }


def _passes_filter(mx):
    """Reject obviously broken takes (tunable thresholds)."""
    return (mx["duration_s"] >= 20.0          # not a stub
            and mx["rms_dbfs"] >= -35.0        # not near-silent
            and mx["clip_ratio"] <= 0.02       # not heavily clipped
            and mx["silence_ratio"] <= 0.6)    # not mostly silence


def _rank_takes(scored):
    """Set rec['score'] for each take. Failed-filter takes get -1.0. Passing
    takes: blend lyric clarity (1-WER) with CLAP style-match, each min-max
    normalised across the song's passing takes so the two scales are comparable.
    When no CLAP value is present, score is the absolute (1-WER)."""
    for r in scored:
        if not r.get("passed"):
            r["score"] = -1.0
    passing = [r for r in scored if r.get("passed")]
    if not passing:
        return
    lyric = [1.0 - (r["wer"] if r.get("wer") is not None else 1.0) for r in passing]
    use_clap = any(r.get("clap_sim") is not None for r in passing)
    if use_clap:
        clap = [r["clap_sim"] if r.get("clap_sim") is not None else 0.0 for r in passing]

        def _mm(xs):
            lo, hi = min(xs), max(xs)
            return [0.5] * len(xs) if hi - lo < 1e-9 else [(x - lo) / (hi - lo) for x in xs]

        ln, cn = _mm(lyric), _mm(clap)
        for r, a, b in zip(passing, ln, cn):
            r["score"] = round(W_LYRIC * a + W_CLAP * b, 4)
    else:
        for r, a in zip(passing, lyric):
            r["score"] = round(a, 4)


if SCORE_VERSIONS:
    try:
        try:
            import whisper
        except ImportError:
            sh(f"{sys.executable} -m pip install -q openai-whisper")
            import whisper
        print(f"[score] loading Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} ...")
        wmodel = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

        # Optional CLAP style-match model (HF transformers). On any failure we
        # fall back to lyric-WER-only ranking rather than aborting scoring.
        clap = clap_proc = None
        if SCORE_CLAP:
            try:
                import torch
                try:
                    from transformers import ClapModel, ClapProcessor
                except ImportError:
                    sh(f"{sys.executable} -m pip install -q transformers")
                    from transformers import ClapModel, ClapProcessor
                print(f"[score] loading CLAP '{CLAP_MODEL}' on {CLAP_DEVICE} ...")
                clap = ClapModel.from_pretrained(CLAP_MODEL).to(CLAP_DEVICE).eval()
                clap_proc = ClapProcessor.from_pretrained(CLAP_MODEL)
            except Exception as exc:
                print(f"[score] CLAP disabled ({type(exc).__name__}: {exc}); lyric-WER only")
                clap = clap_proc = None

        def _clap_sim(wav48k, text):
            """Cosine similarity between the caption text and the audio (48 kHz)."""
            import torch
            with torch.no_grad():
                ti = clap_proc(text=[text], return_tensors="pt", padding=True).to(CLAP_DEVICE)
                ai = clap_proc(audios=wav48k, sampling_rate=48000,
                               return_tensors="pt").to(CLAP_DEVICE)
                te = clap.get_text_features(**ti)
                ae = clap.get_audio_features(**ai)
                te = te / te.norm(dim=-1, keepdim=True)
                ae = ae / ae.norm(dim=-1, keepdim=True)
                return float((te * ae).sum(dim=-1).item())

        for m in manifest:
            song = songs[m["index"] - 1] if 0 <= m["index"] - 1 < len(songs) else {}
            ref_words = _norm_words(song.get("lyrics", ""))
            caption = song.get("prompt") or song.get("caption") or ""
            scored = []
            for ver in m.get("versions", []):
                for path in ver.get("paths", []):
                    try:
                        wav = whisper.load_audio(path)            # 16 kHz mono float32
                        mx = _audio_metrics(wav, 16000)
                        passed = _passes_filter(mx)
                        tr = wmodel.transcribe(wav, fp16=(WHISPER_DEVICE != "cpu"))
                        wer = round(_wer(ref_words, _norm_words(tr.get("text", ""))), 4)
                        clap_sim = None
                        if clap is not None and caption:
                            try:
                                clap_sim = round(_clap_sim(
                                    whisper.load_audio(path, sr=48000), caption), 4)
                            except Exception as exc:
                                print(f"   [score] clap failed v{ver.get('version')}: {exc}")
                        rec = {"version": ver.get("version"), "path": path,
                               "passed": passed, "wer": wer, "clap_sim": clap_sim,
                               "metrics": mx,
                               "transcript": (tr.get("text") or "").strip()[:500]}
                    except Exception as exc:
                        rec = {"version": ver.get("version"), "path": path,
                               "passed": False, "wer": None, "clap_sim": None,
                               "error": f"{type(exc).__name__}: {exc}"}
                    scored.append(rec)
                    print(f"   [score] v{rec['version']} pass={rec['passed']} "
                          f"wer={rec.get('wer')} clap={rec.get('clap_sim')} "
                          f"{os.path.basename(path)}")

            _rank_takes(scored)                       # sets rec['score'] for all takes
            m["scores"] = scored
            ranked = sorted(scored, key=lambda r: r.get("score", -1.0), reverse=True)
            if ranked:
                best = ranked[0]
                orig = best["path"]
                m["best_version"], m["best_wer"] = best["version"], best.get("wer")
                m["best_clap"], m["best_path"] = best.get("clap_sim"), orig
                try:                                  # mark the winner so it stands out in Drive
                    p = pathlib.Path(orig)
                    if p.exists():
                        bp = str(p.with_name(p.stem + "_BEST" + p.suffix))
                        os.rename(orig, bp)
                        best["path"] = m["best_path"] = bp
                        for ver in m["versions"]:
                            ver["paths"] = [bp if x == orig else x for x in ver.get("paths", [])]
                except Exception as exc:
                    print(f"[score] (could not rename winner: {exc})")
                print(f"[score] BEST {m['title']!r}: v{m['best_version']} "
                      f"wer={m['best_wer']} clap={m['best_clap']} score={best.get('score')} "
                      f"-> {os.path.basename(m['best_path'])}")
    except Exception as exc:
        print(f"[score] scoring skipped ({type(exc).__name__}: {exc})")

# Manifest
with open(os.path.join(LOCAL_OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nmanifest -> {LOCAL_OUT}/manifest.json")

# 10) Google Drive upload. Auth is auto-detected from the Kaggle Secrets present:
#   GDRIVE_OAUTH_TOKEN -> personal "My Drive" (OAuth authorized-user token).
#   GDRIVE_SA_JSON     -> Shared Drive (service account). Used only if no OAuth
#                         token is set, so existing Shared-Drive setups keep working.
# GDRIVE_FOLDER_ID (constant above or same-named secret) is the destination folder.
if OUTPUT_MODE == "gdrive":
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()

    def _secret(name):
        try:
            return _secrets.get_secret(name)
        except Exception:
            return ""

    # Personal My Drive via OAuth refresh token. Supply EITHER:
    #   GDRIVE_OAUTH_TOKEN  -> one authorized-user JSON (client_id + client_secret
    #                          + refresh_token), as printed by gdrive_oauth_setup.py
    # OR the three pieces separately (handy with the Google OAuth Playground):
    #   GDRIVE_REFRESH_TOKEN + GDRIVE_CLIENT_ID + GDRIVE_CLIENT_SECRET
    oauth_token   = _secret("GDRIVE_OAUTH_TOKEN")
    refresh_token = _secret("GDRIVE_REFRESH_TOKEN")
    client_id     = _secret("GDRIVE_CLIENT_ID")
    client_secret = _secret("GDRIVE_CLIENT_SECRET")
    have_oauth = bool(oauth_token) or bool(refresh_token and client_id and client_secret)
    sa_json    = "" if have_oauth else _secret("GDRIVE_SA_JSON")
    if not have_oauth and not sa_json:
        raise RuntimeError(
            "Drive upload needs Kaggle Secrets: GDRIVE_OAUTH_TOKEN, or "
            "GDRIVE_REFRESH_TOKEN + GDRIVE_CLIENT_ID + GDRIVE_CLIENT_SECRET "
            "(personal My Drive), or GDRIVE_SA_JSON (Shared Drive).")

    folder = GDRIVE_FOLDER_ID or _secret("GDRIVE_FOLDER_ID")
    if not folder:
        raise RuntimeError("Set GDRIVE_FOLDER_ID (constant above or Kaggle Secret).")

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
        from google.auth.transport.requests import Request as GoogleAuthRequest
    except ImportError:
        sh(f"{sys.executable} -m pip install -q "
           f"google-api-python-client google-auth google-auth-oauthlib")
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
        from google.auth.transport.requests import Request as GoogleAuthRequest

    # Full drive scope (not drive.file) so the token may upload into a
    # pre-existing folder it did not itself create.
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    if have_oauth:
        if oauth_token:
            info = json.loads(oauth_token)
        else:
            info = {
                "refresh_token": refresh_token,
                "client_id":     client_id,
                "client_secret": client_secret,
                "token_uri":     "https://oauth2.googleapis.com/token",
                "type":          "authorized_user",
            }
        creds = UserCredentials.from_authorized_user_info(info, scopes=SCOPES)
        # The stored creds carry only the refresh token; mint a live access token.
        if not creds.valid and creds.refresh_token:
            creds.refresh(GoogleAuthRequest())
        print("[drive] auth: OAuth user refresh token (personal My Drive)")
    else:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(sa_json), scopes=SCOPES)
        print("[drive] auth: service account (Shared Drive)")
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    AUDIO_EXTS = {".mp3", ".wav", ".flac", ".opus", ".aac"}
    uploaded = 0
    for p in pathlib.Path(LOCAL_OUT).rglob("*"):
        if p.is_file() and (p.suffix.lower() in AUDIO_EXTS or p.name == "manifest.json"):
            meta = {"name": p.name, "parents": [folder]}
            res = svc.files().create(
                body=meta,
                media_body=MediaFileUpload(str(p), resumable=True),
                fields="id, webViewLink", supportsAllDrives=True).execute()
            link = res.get("webViewLink") or res.get("id")
            print(f"[drive] {p.name} -> {link}")
            uploaded += 1
    print(f"[drive] {uploaded} file(s) uploaded.")
else:
    print(f"[local] outputs in {LOCAL_OUT}")
'''


def build_notebook(out_path: str) -> None:
    nb = {
        "cells": [{
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": NOTEBOOK_CELL_TEMPLATE,
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Wrote {out_path} (1 cell)")


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "acestep_kaggle_2xT4_weekly.ipynb",
    )
    build_notebook(out)

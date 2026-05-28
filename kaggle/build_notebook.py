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
    or uploaded to a Shared Drive folder when OUTPUT_MODE="gdrive"

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
#   - Optional Drive upload: add Kaggle Secret GDRIVE_SA_JSON (service account
#     JSON). The target folder must live in a SHARED DRIVE (service accounts
#     have no personal-Drive quota). Put the folder id in GDRIVE_FOLDER_ID
#     below or expose it as the Kaggle secret GDRIVE_FOLDER_ID.
# =============================================================================

# ----- CONFIG (edit these) ---------------------------------------------------
REPO_URL    = "https://github.com/hardik2015/ACE-Step-1.5.git"
REPO_BRANCH = "main"
WORK_DIR    = "/kaggle/tmp/ACE-Step-1.5"

OUTPUT_MODE       = "local"                                # "local" | "gdrive"
LOCAL_OUT         = "/kaggle/working/acestep_output"
GDRIVE_FOLDER_ID  = ""                                     # Shared Drive folder id

INPUT_MODE        = "inline"                               # "inline" | "file"
INPUT_JSON_PATH   = "/kaggle/input/your-dataset/songs.json"

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
STABILITY_ENV = {
    "ACESTEP_CONFIG_PATH":         "acestep-v15-sft",
    "ACESTEP_DTYPE":               "float32",
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

# ----- IMPLEMENTATION --------------------------------------------------------
import os, sys, json, time, shutil, subprocess, pathlib, datetime
import urllib.request
import urllib.parse


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
           "--lm-model-path", "acestep-5Hz-lm-0.6B"]
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
def _api_post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _api_download(server_path, dest_path):
    """Pull audio bytes via /v1/audio (server path-scoped to temp dir)."""
    qs = urllib.parse.urlencode({"path": server_path})
    with urllib.request.urlopen(f"{API_BASE}/v1/audio?{qs}", timeout=120) as r, \
         open(dest_path, "wb") as out:
        shutil.copyfileobj(r, out)


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
        if k == "title":
            continue
        body[aliases.get(k, k)] = v
    return body


def _wait_for_job(task_id, deadline):
    """Poll /query_result until the job leaves status 0. Returns the parsed
    result list (a list of dicts with 'file', 'metas', etc.)."""
    last_progress = ""
    while time.time() < deadline:
        resp = _api_post("/query_result", {"task_id_list": [task_id]})
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
            raise RuntimeError(f"job {task_id} failed: {item.get('result')}")
        time.sleep(JOB_POLL_INTERVAL_S)
    raise TimeoutError(f"job {task_id} did not finish within {JOB_TIMEOUT_S}s")


# 9) Run generation, one song at a time, on the same shared init.
os.makedirs(LOCAL_OUT, exist_ok=True)
manifest = []
for i, song in enumerate(songs, 1):
    title = (song.get("title") or song.get("prompt") or song.get("caption") or f"song{i}")[:80]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\n[gen {i}/{len(songs)}] {title!r}")
    body = _build_request_body(song)

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
        server_path = item.get("file") or ""
        if not server_path:
            continue
        ext = pathlib.Path(server_path).suffix or f".{GEN_DEFAULTS['audio_format']}"
        dst = os.path.join(LOCAL_OUT, f"{stamp}_{i:03d}_{ai}{ext}")
        if os.path.exists(server_path):
            # API server runs in the same container, so the file is reachable.
            shutil.copy2(server_path, dst)
        else:
            _api_download(server_path, dst)
        print(f"   -> {dst} ({os.path.getsize(dst)//1024} KB)")
        saved.append(dst)
    manifest.append({
        "index": i,
        "title": title,
        "task_id": task_id,
        "paths": saved,
        "success": bool(saved),
    })

# Manifest
with open(os.path.join(LOCAL_OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nmanifest -> {LOCAL_OUT}/manifest.json")

# 10) Optional Google Drive upload (Shared Drive folder required for SA).
if OUTPUT_MODE == "gdrive":
    try:
        from kaggle_secrets import UserSecretsClient
        sa_json = UserSecretsClient().get_secret("GDRIVE_SA_JSON")
    except Exception as exc:
        raise RuntimeError(f"Need Kaggle Secret GDRIVE_SA_JSON for Drive upload: {exc}")
    folder = GDRIVE_FOLDER_ID
    if not folder:
        try:
            folder = UserSecretsClient().get_secret("GDRIVE_FOLDER_ID")
        except Exception:
            raise RuntimeError("Set GDRIVE_FOLDER_ID or a Kaggle Secret of the same name.")
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        sh(f"{sys.executable} -m pip install -q google-api-python-client google-auth")
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from googleapiclient.http import MediaFileUpload

    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json), scopes=["https://www.googleapis.com/auth/drive"])
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

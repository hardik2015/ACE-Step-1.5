"""Generate the ACE-Step 2x T4 weekly Kaggle notebook (.ipynb).

Run:  python kaggle/build_notebook.py
Produces: kaggle/acestep_kaggle_2xT4_weekly.ipynb
"""
import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src,
        }
    )


md(r"""# ACE-Step 1.5 — 2× T4 weekly generator (Kaggle)

Runs **ACE-Step** headless on Kaggle's **GPU T4 ×2 (2×16GB)** with a split layout:

* **GPU 0 (`cuda:0`)** — DiT (`acestep-v15-sft`) + VAE + text-encoder, no CPU offload.
* **GPU 1 (`cuda:1`)** — the **4B 5Hz LM** (PyTorch backend).

It takes a **song JSON** (a single song, a list, or `{"songs": [...]}`) — either pasted
into a variable for testing or read from a file — generates audio, and writes the
`.mp3` + `.json` result to a **local directory** or **Google Drive**.

## Before you run
1. **Settings → Accelerator → `GPU T4 x2`**.
2. **Settings → Internet → On** (needed to clone the repo and download model weights).
3. *(Google Drive output only)* add Kaggle **Secrets** (Add-ons → Secrets):
   * `GDRIVE_SA_JSON` — a Google **service-account** JSON key (Drive API enabled).
   * Share your target Drive folder with the service-account email (Editor), and put
     the folder ID in `GDRIVE_FOLDER_ID` below (or as a secret of the same name).

## Run weekly on a schedule
Save Version → **Save & Run All (Commit)** → enable **Schedule**, pick **Weekly**.
Scheduled runs are headless and must finish within 12h, so keep the song list small.
Everything in this notebook runs top-to-bottom with no interactive input.
""")

md("## 1. Configuration — edit this cell")

code(r'''# ----------------------------- Repo -----------------------------
REPO_URL    = "https://github.com/hardik2015/ACE-Step-1.5.git"
REPO_BRANCH = "main"
WORKDIR     = "/kaggle/working/ACE-Step-1.5"

# --------------------------- Output -----------------------------
# "local"  -> save under LOCAL_OUTPUT_DIR (persisted in /kaggle/working)
# "gdrive" -> upload to a Google Drive folder via a service account (see header)
OUTPUT_MODE       = "local"
LOCAL_OUTPUT_DIR  = "/kaggle/working/acestep_output"
GDRIVE_FOLDER_ID  = ""          # Drive folder ID (or set a Kaggle secret of this name)

# ---------------------------- Input ----------------------------
# "inline" -> use SONG_JSON_INLINE below (handy for testing)
# "file"   -> read INPUT_JSON_PATH (e.g. a file from an attached Kaggle dataset)
INPUT_MODE      = "inline"
INPUT_JSON_PATH = "/kaggle/input/your-dataset/songs.json"

# Paste one song, a list of songs, or {"songs": [ ... ]}.
# Any field accepted by POST /release_task works; unspecified fields fall back to
# GEN_DEFAULTS below. See the project's docs/en/API.md for the full schema.
SONG_JSON_INLINE = r"""
{
  "prompt": "warm indie pop, female vocal, gentle piano, soft drums",
  "lyrics": "[verse]\nMorning light across the room\n[chorus]\nAnd we rise, and we rise\n",
  "audio_duration": 120,
  "bpm": 96,
  "key_scale": "C major",
  "vocal_language": "en"
}
"""

# Defaults merged UNDER each song (the song's own fields win).
GEN_DEFAULTS = {
    "thinking": True,          # use the 5Hz LM (runs on GPU1) — higher quality
    "batch_size": 1,           # keep modest: DiT runs on a real 16GB GPU0
    "audio_format": "mp3",     # mp3 | wav | flac | opus | aac | wav32
    "vocal_language": "en",
    "inference_steps": 50,     # sft/base model: ~32-64. For a turbo model use 8.
    "guidance_scale": 7.0,     # effective for base/sft models
    "use_random_seed": True,
}

# --------------------- Device / model layout --------------------
# These drive the 2x T4 split. Treating the box as "24GB class" (MAX_CUDA_VRAM=24)
# is what lets the tiering logic enable the 4B LM and disable CPU offload; the LM
# physically lives on GPU1, and GPU0's real 16GB comfortably holds DiT+VAE+enc at
# the small batch size above.
ENV = {
    "ACESTEP_CONFIG_PATH":       "acestep-v15-sft",
    "ACESTEP_LM_MODEL_PATH":     "acestep-5Hz-lm-4B",
    "ACESTEP_DEVICE":            "cuda",      # DiT/VAE/text-encoder -> GPU0
    "ACESTEP_LM_DEVICE":         "cuda:1",    # 5Hz LM             -> GPU1
    "ACESTEP_LM_BACKEND":        "pt",        # cuda:1 forces the PyTorch backend anyway
    "ACESTEP_INIT_LLM":          "true",
    "ACESTEP_OFFLOAD_TO_CPU":    "false",
    "ACESTEP_OFFLOAD_DIT_TO_CPU":"false",
    "ACESTEP_NO_INIT":           "false",     # eager-load models at server startup
    "MAX_CUDA_VRAM":             "24",        # tier6b: 4B allowed + no offload
    "ACESTEP_DTYPE":             "float32",   # T4 (Turing) has no bf16; fp16 overflows
                                              # to NaN latents — force fp32 for the DiT.
}

# --------------------------- Server ----------------------------
API_HOST = "127.0.0.1"
API_PORT = 8001

# How long to wait for the server to come up (model download + load), and for a
# single generation task to finish.
SERVER_READY_TIMEOUT_S = 60 * 40    # first run downloads several GB of weights
POLL_TIMEOUT_S         = 60 * 30
print("Config loaded.")
''')

md("## 2. Apply environment + show the two GPUs")

code(r'''import os, subprocess

for k, v in ENV.items():
    os.environ[k] = v

print("Environment overrides:")
for k in ENV:
    print(f"  {k} = {os.environ[k]}")

print("\nGPUs visible to this notebook:")
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True).stdout or "(nvidia-smi unavailable)")
''')

md("## 3. Clone the repo")

code(r'''import os, subprocess

def run(cmd, **kw):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")

if not os.path.isdir(os.path.join(WORKDIR, ".git")):
    run(["git", "clone", "--branch", REPO_BRANCH, "--depth", "1", REPO_URL, WORKDIR])
else:
    run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", REPO_BRANCH])
    run(["git", "-C", WORKDIR, "checkout", REPO_BRANCH])
    run(["git", "-C", WORKDIR, "reset", "--hard", f"origin/{REPO_BRANCH}"])

os.chdir(WORKDIR)
print("Repo ready at", WORKDIR)
''')

md(r"""## 4. Install dependencies

This keeps **Kaggle's preinstalled PyTorch** (matched to the T4 driver) and installs
ACE-Step with `--no-deps`, then layers the non-torch runtime libraries on top. The
first run takes several minutes. If a generation step later complains about a missing
module, add it to `EXTRA_DEPS` and re-run this cell.
""")

code(r'''import sys, subprocess

def pip(*args):
    print("$ pip", " ".join(args))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])

import torch
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda,
      "| GPUs:", torch.cuda.device_count())

# 1) Register the `acestep` / `acestep-api` entry points without disturbing torch.
pip("-e", ".", "--no-deps")

# 2) Runtime libraries the API + PyTorch-backend LM path need (no torch here).
EXTRA_DEPS = [
    "transformers>=4.51.0,<4.58.0", "diffusers>=0.37.0", "accelerate>=1.12.0",
    "fastapi>=0.110.0", "uvicorn[standard]>=0.27.0", "loguru>=0.7.3",
    "einops>=0.8.1", "soundfile>=0.13.1", "scipy>=1.10.1", "diskcache",
    "vector-quantize-pytorch>=1.27.15", "numba>=0.63.1", "toml",
    "pytorch-wavelets>=1.3.0", "pywavelets>=1.9.0", "modelscope",
    "typer-slim>=0.21.1", "peft>=0.18.0", "huggingface_hub",
]
pip(*EXTRA_DEPS)

# 3) Best-effort extras (match the resident torch; harmless if they fail).
for spec in ["torchao", "torchcodec"]:
    try:
        pip(spec)
    except Exception as e:
        print(f"[warn] could not install {spec}: {e}")

# 4) Vendored nano-vllm — only used by the vllm backend, installed for safe imports.
try:
    pip("-e", "acestep/third_parts/nano-vllm", "--no-deps")
except Exception as e:
    print(f"[warn] nano-vllm install skipped: {e}")

print("Dependency install step finished.")
''')

md(r"""## 5. Launch the API server (background) and wait until ready

`ACESTEP_NO_INIT=false` makes the server download + load all weights before it starts
serving, so once `/health` answers, generation is ready. Logs stream to
`/kaggle/working/acestep_api.log`.
""")

code(r'''import subprocess, time, os, requests, pathlib

API = f"http://{API_HOST}:{API_PORT}"
LOG_PATH = "/kaggle/working/acestep_api.log"

# Reuse an already-running server if this cell is re-run.
def _alive():
    try:
        return requests.get(f"{API}/health", timeout=5).status_code == 200
    except Exception:
        return False

if _alive():
    print("Server already running.")
else:
    log = open(LOG_PATH, "w")
    proc = subprocess.Popen(
        ["acestep-api", "--host", API_HOST, "--port", str(API_PORT)],
        stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), cwd=WORKDIR,
    )
    print(f"Launched acestep-api (pid {proc.pid}); waiting for /health ...")
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            print("Server process exited early. Last log lines:")
            print("".join(pathlib.Path(LOG_PATH).read_text().splitlines(keepends=True)[-40:]))
            raise RuntimeError("acestep-api failed to start")
        if _alive():
            ready = True
            break
        time.sleep(5)
    if not ready:
        raise TimeoutError("Server did not become ready in time; check the log.")

h = requests.get(f"{API}/health", timeout=10).json()["data"]
print("Health:", h)
print(f"  models_initialized={h.get('models_initialized')} "
      f"loaded_model={h.get('loaded_model')} "
      f"llm_initialized={h.get('llm_initialized')} "
      f"loaded_lm_model={h.get('loaded_lm_model')}")
''')

md("## 6. Helpers — load input, submit, poll, save (local or Google Drive)")

code(r'''import json, os, time, datetime, requests

API = f"http://{API_HOST}:{API_PORT}"

def load_songs():
    if INPUT_MODE == "file":
        with open(INPUT_JSON_PATH) as f:
            data = json.load(f)
    else:
        data = json.loads(SONG_JSON_INLINE)
    if isinstance(data, dict):
        if isinstance(data.get("songs"), list):
            return data["songs"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be an object, a list, or {'songs': [...]}.")

def submit(song):
    payload = {**GEN_DEFAULTS, **song}
    r = requests.post(f"{API}/release_task", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["task_id"], payload

def poll(task_id, timeout=POLL_TIMEOUT_S, interval=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.post(f"{API}/query_result",
                          json={"task_id_list": [task_id]}, timeout=60)
        r.raise_for_status()
        items = r.json().get("data") or []
        if items:
            it = items[0]
            status = it.get("status")
            if status == 1:
                res = it.get("result")
                return json.loads(res) if isinstance(res, str) else res
            if status == 2:
                raise RuntimeError(f"Task {task_id} failed: {it}")
        time.sleep(interval)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

def download(file_url):
    r = requests.get(f"{API}{file_url}", timeout=600)
    r.raise_for_status()
    return r.content

_EXT = {"wav32": "wav"}
def _ext(fmt):
    return _EXT.get(fmt, fmt or "mp3")

# ---- Google Drive (service account; headless-safe) ----
_drive = None
def _get_drive():
    global _drive
    if _drive is not None:
        return _drive
    sa_json = None
    try:
        from kaggle_secrets import UserSecretsClient
        sa_json = UserSecretsClient().get_secret("GDRIVE_SA_JSON")
    except Exception as e:
        raise RuntimeError(f"Could not read GDRIVE_SA_JSON secret: {e}")
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except Exception:
        import sys, subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "google-api-python-client", "google-auth"])
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json), scopes=["https://www.googleapis.com/auth/drive"])
    _drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _drive

def _gdrive_folder_id():
    if GDRIVE_FOLDER_ID:
        return GDRIVE_FOLDER_ID
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GDRIVE_FOLDER_ID")
    except Exception:
        raise RuntimeError("Set GDRIVE_FOLDER_ID (variable or Kaggle secret).")

def _gdrive_upload(local_path, name):
    from googleapiclient.http import MediaFileUpload
    svc = _get_drive()
    meta = {"name": name, "parents": [_gdrive_folder_id()]}
    media = MediaFileUpload(local_path, resumable=True)
    f = svc.files().create(body=meta, media_body=media, fields="id, webViewLink",
                           supportsAllDrives=True).execute()
    return f.get("webViewLink") or f.get("id")

def save_output(stamp, task_id, idx, audio_bytes, item, fmt):
    """Write audio+json locally, then mirror to Drive if OUTPUT_MODE == 'gdrive'."""
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    base = f"{stamp}_{task_id[:8]}_{idx}"
    apath = os.path.join(LOCAL_OUTPUT_DIR, f"{base}.{_ext(fmt)}")
    jpath = os.path.join(LOCAL_OUTPUT_DIR, f"{base}.json")
    with open(apath, "wb") as f:
        f.write(audio_bytes)
    with open(jpath, "w") as f:
        json.dump(item, f, indent=2, ensure_ascii=False)
    dest = apath
    if OUTPUT_MODE == "gdrive":
        link = _gdrive_upload(apath, os.path.basename(apath))
        _gdrive_upload(jpath, os.path.basename(jpath))
        dest = link
    return apath, dest

print("Helpers ready.")
''')

md("## 7. Generate")

code(r'''import datetime, json

songs = load_songs()
print(f"Loaded {len(songs)} song(s). Output mode: {OUTPUT_MODE}\n")

manifest = []
for si, song in enumerate(songs, 1):
    title = (song.get("prompt") or song.get("sample_query") or "song")[:60]
    print(f"[{si}/{len(songs)}] submitting: {title!r}")
    task_id, payload = submit(song)
    print(f"    task_id={task_id} — waiting...")
    results = poll(task_id)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fmt = payload.get("audio_format", "mp3")
    for ai, item in enumerate(results, 1):
        file_url = item.get("file")
        if not file_url:
            print(f"    [{ai}] no file in result, skipping")
            continue
        audio = download(file_url)
        local_path, dest = save_output(stamp, task_id, ai, audio, item, fmt)
        size_kb = len(audio) // 1024
        print(f"    [{ai}] saved {size_kb} KB -> {dest}")
        manifest.append({"song_index": si, "task_id": task_id,
                         "local_path": local_path, "destination": dest,
                         "bytes": len(audio)})

print("\nDone. Manifest:")
print(json.dumps(manifest, indent=2))

with open("/kaggle/working/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("\nManifest written to /kaggle/working/manifest.json")
''')

md(r"""## Notes & troubleshooting

* **The 4B LM downgrade:** the API server normally downgrades the 4B LM to 1.7B on a
  16GB GPU. `MAX_CUDA_VRAM=24` (set in cell 1) makes the tiering treat the machine as
  24GB-class so the 4B is kept and CPU offload is disabled — valid here because the LM
  actually lives on the second GPU. Keep `batch_size` small so GPU0's real 16GB isn't
  exceeded.
* **T4 has no bf16 tensor cores** (Turing). The code runs bf16, which works but isn't
  as fast as fp16 — expect a few minutes per song with the 4B LM + `thinking=true`.
* **Turbo model instead of sft:** set `ACESTEP_CONFIG_PATH=acestep-v15-turbo` and drop
  `inference_steps` to 8 in `GEN_DEFAULTS`.
* **Avoid re-downloading weights every week:** attach the checkpoints as a Kaggle
  dataset and point `ACESTEP_CHECKPOINTS_DIR` at the mounted path in `ENV`.
* **Server log:** `/kaggle/working/acestep_api.log`.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "acestep_kaggle_2xT4_weekly.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Wrote", out, "with", len(cells), "cells")

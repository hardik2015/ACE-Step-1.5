"""Generate the ACE-Step single-cell Kaggle notebook (.ipynb).

The output notebook has ONE code cell that:
  - applies the stability env from the working ace-step.ipynb
  - clones the user's fork to /kaggle/tmp, runs `uv sync`
  - launches the SAME `acestep` Gradio UI you've been running (in the background)
  - connects to it with `gradio_client` and calls the EXACT click-handler the
    Generate button triggers, with the 78 positional inputs in the order
    declared in `generation_run_wiring.py`
  - per-song fields override Gradio defaults; everything else uses Gradio's
    own defaults (mirroring exactly what your "Generate" click sends)
  - audio path returned by Gradio is copied to LOCAL_OUT (or uploaded to a
    Shared Drive folder when OUTPUT_MODE="gdrive")

This is the cleanest possible approach: use the Gradio path that already
produces music, just drive its button programmatically.

Run:  python kaggle/build_notebook.py
Output: kaggle/acestep_kaggle_2xT4_weekly.ipynb
"""
import json
import os


# The exact ordered input list bound to the Generate button in
# acestep/ui/gradio/events/wiring/generation_run_wiring.py (.then chain step 2).
# DO NOT REORDER — these match Gradio's positional input contract 1:1.
INPUT_ORDER = [
    "captions", "lyrics", "bpm", "key_scale", "time_signature", "vocal_language",
    "inference_steps", "guidance_scale", "random_seed_checkbox", "seed",
    "reference_audio", "audio_duration", "batch_size_input", "src_audio",
    "text2music_audio_code_string", "repainting_start", "repainting_end",
    "instruction_display_gen", "audio_cover_strength", "cover_noise_strength",
    "task_type", "no_fsq", "use_adg", "cfg_interval_start", "cfg_interval_end",
    "shift", "infer_method", "sampler_mode", "velocity_norm_threshold",
    "velocity_ema_factor", "dcw_enabled", "dcw_mode", "dcw_scaler",
    "dcw_high_scaler", "dcw_wavelet", "custom_timesteps", "audio_format",
    "mp3_bitrate", "mp3_sample_rate", "lm_temperature", "think_checkbox",
    "lm_cfg_scale", "lm_top_k", "lm_top_p", "lm_negative_prompt",
    "use_cot_metas", "use_cot_caption", "use_cot_language",
    "is_format_caption_state", "constrained_decoding_debug", "allow_lm_batch",
    "auto_score", "auto_lrc", "score_scale", "lm_batch_chunk_size",
    "track_name", "complete_track_classes", "enable_normalization",
    "normalization_db", "fade_in_duration", "fade_out_duration",
    "latent_shift", "latent_rescale", "repaint_mode", "repaint_strength",
    "retake_variance", "retake_seed", "flow_edit_morph",
    "flow_edit_source_caption", "flow_edit_source_lyrics",
    "flow_edit_n_min", "flow_edit_n_max", "flow_edit_n_avg",
    "autogen_checkbox", "current_batch_index", "total_batches",
    "batch_queue", "generation_params_state",
]

# Per-component defaults mirroring the Gradio UI defaults (sourced from
# GenerationParams / GenerationConfig dataclasses plus the working demo log).
UI_DEFAULTS = {
    "captions": "",
    "lyrics": "",
    "bpm": None,
    "key_scale": "",
    "time_signature": "",
    "vocal_language": "unknown",            # what produced music in the working demo
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "random_seed_checkbox": True,           # use_random_seed
    "seed": -1,
    "reference_audio": None,
    "audio_duration": -1.0,                 # -1 -> LM picks (consistent codes/latents)
    "batch_size_input": 1,
    "src_audio": None,
    "text2music_audio_code_string": "",
    "repainting_start": 0.0,
    "repainting_end": -1.0,
    "instruction_display_gen": "Fill the audio semantic mask based on the given conditions:",
    "audio_cover_strength": 1.0,
    "cover_noise_strength": 0.0,
    "task_type": "text2music",
    "no_fsq": False,
    "use_adg": False,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "shift": 1.0,
    "infer_method": "ode",
    "sampler_mode": "euler",
    "velocity_norm_threshold": 0.0,
    "velocity_ema_factor": 0.0,
    "dcw_enabled": True,
    "dcw_mode": "double",
    "dcw_scaler": 0.05,
    "dcw_high_scaler": 0.02,
    "dcw_wavelet": "haar",
    "custom_timesteps": "",
    "audio_format": "mp3",
    "mp3_bitrate": "128k",
    "mp3_sample_rate": 48000,
    "lm_temperature": 0.85,
    "think_checkbox": True,                 # thinking=True
    "lm_cfg_scale": 2.0,
    "lm_top_k": 0,
    "lm_top_p": 0.9,
    "lm_negative_prompt": "NO USER INPUT",
    "use_cot_metas": True,
    "use_cot_caption": False,               # KEY: no LM caption rewrite
    "use_cot_language": True,               # MATCHES the working Gradio run
    "is_format_caption_state": False,
    "constrained_decoding_debug": False,
    "allow_lm_batch": True,
    "auto_score": False,
    "auto_lrc": False,
    "score_scale": 1.0,
    "lm_batch_chunk_size": 8,
    "track_name": "",
    "complete_track_classes": "",
    "enable_normalization": True,
    "normalization_db": -1.0,
    "fade_in_duration": 0.0,
    "fade_out_duration": 0.0,
    "latent_shift": 0.0,
    "latent_rescale": 1.0,
    "repaint_mode": "balanced",
    "repaint_strength": 0.5,
    "retake_variance": 0.0,
    "retake_seed": None,
    "flow_edit_morph": False,
    "flow_edit_source_caption": "",
    "flow_edit_source_lyrics": "",
    "flow_edit_n_min": 0.0,
    "flow_edit_n_max": 1.0,
    "flow_edit_n_avg": 1,
    "autogen_checkbox": False,
    "current_batch_index": 0,
    "total_batches": 1,
    "batch_queue": [],
    "generation_params_state": None,
}

# Aliases the song JSON may use -> the UI field name in INPUT_ORDER.
SONG_ALIASES = {
    "prompt": "captions",
    "caption": "captions",
    "keyscale": "key_scale",
    "timesignature": "time_signature",
    "duration": "audio_duration",
    "batch_size": "batch_size_input",
    "thinking": "think_checkbox",
    "use_random_seed": "random_seed_checkbox",
}


# ---------------------------------------------------------------------------
# The notebook cell — orchestration only.
# ---------------------------------------------------------------------------
NOTEBOOK_CELL_TEMPLATE = r'''# =============================================================================
# ACE-Step 2x T4 — single-cell headless generator (Gradio-API driven)
#
# Launches the SAME Gradio UI you already use successfully, then drives its
# Generate button via gradio_client (no UI clicks, no in-process reimpl).
# Per-song fields override Gradio defaults; everything else mirrors what the
# UI sends when you click Generate.
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

# Defaults merged UNDER each song. Field names match the song-JSON API style.
GEN_DEFAULTS = {
    "thinking":          True,
    "use_cot_caption":   False,    # KEY: keeps YOUR caption
    "use_cot_language":  True,     # matches the working Gradio run
    "use_cot_metas":     True,
    "inference_steps":   50,
    "guidance_scale":    7.0,
    "use_random_seed":   True,
    "audio_format":      "mp3",
    "batch_size":        1,
    # No "duration" => the LM picks (consistent codes/latents -> clean audio).
}

# The Gradio API endpoint name. The Generate button wires three handlers via
# .click(...).then(...).then(...); the second one (the actual generator) is
# what we want. The notebook prints view_api() so you can confirm the name.
GRADIO_API_NAME = None        # e.g. "/lambda_1"; None => auto-detect by signature

# Multi-GPU layout (matches the working ace-step.ipynb).
STABILITY_ENV = {
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

GRADIO_PORT = 7860
GRADIO_LOG  = "/kaggle/working/gradio.log"
GRADIO_READY_TIMEOUT_S = 60 * 30           # first run downloads ~13GB weights

# ----- HARD-CODED UI CONTRACT (from generation_run_wiring.py) ----------------
INPUT_ORDER   = __INPUT_ORDER__
UI_DEFAULTS   = __UI_DEFAULTS__
SONG_ALIASES  = __SONG_ALIASES__

# ----- IMPLEMENTATION --------------------------------------------------------
import os, sys, json, time, shutil, subprocess, pathlib, datetime, shlex
import urllib.request


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

# 6) Install gradio_client in the Kaggle notebook process (we drive Gradio from
#    here over HTTP -- no need to import acestep here).
sh(f"{sys.executable} -m pip install -q 'gradio_client>=1.0.0'")
from gradio_client import Client

# 7) Launch Gradio in the background using the project .venv (same command as
#    your working ace-step.ipynb, without ngrok).
def _gradio_alive():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{GRADIO_PORT}/config", timeout=5)
        return True
    except Exception:
        return False

if not _gradio_alive():
    log_f = open(GRADIO_LOG, "w")
    cmd = ["uv", "run", "acestep",
           "--device", "cuda", "--backend", "pt",
           "--lm_device", "cuda:1",
           "--server-name", "127.0.0.1",
           "--port", str(GRADIO_PORT)]
    print("launching:", " ".join(cmd), f"(logs -> {GRADIO_LOG})")
    proc = subprocess.Popen(cmd, cwd=WORK_DIR, env=os.environ.copy(),
                            stdout=log_f, stderr=subprocess.STDOUT)
    print(f"gradio pid={proc.pid}, waiting up to {GRADIO_READY_TIMEOUT_S}s for /config ...")
    deadline = time.time() + GRADIO_READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            print("gradio exited early. Tail of log:")
            with open(GRADIO_LOG) as f:
                print("".join(f.readlines()[-50:]))
            raise RuntimeError("gradio failed to start")
        if _gradio_alive():
            break
        time.sleep(5)
    else:
        raise TimeoutError("gradio not ready in time")
print(f"gradio ready on http://127.0.0.1:{GRADIO_PORT}")

# 8) Connect with gradio_client and print the discovered endpoints.
client = Client(f"http://127.0.0.1:{GRADIO_PORT}", verbose=False)
api = client.view_api(return_format="dict")
print("=== Gradio API endpoints (named_endpoints) ===")
named = (api or {}).get("named_endpoints", {})
for name, spec in named.items():
    n_params = len(spec.get("parameters", []))
    n_out = len(spec.get("returns", []))
    print(f"  {name}: parameters={n_params}, returns={n_out}")

def _autodetect_endpoint(api_dict, expected_params=78):
    """Find the endpoint whose parameter count matches the Generate handler."""
    candidates = []
    for name, spec in (api_dict or {}).get("named_endpoints", {}).items():
        n = len(spec.get("parameters", []))
        n_out = len(spec.get("returns", []))
        candidates.append((name, n, n_out))
    # Prefer exact match on parameter count; then "closest above 50".
    exact = [c for c in candidates if c[1] == expected_params]
    if exact:
        return exact[0][0]
    candidates.sort(key=lambda c: (abs(c[1] - expected_params), -c[1]))
    return candidates[0][0] if candidates else None

target_endpoint = GRADIO_API_NAME or _autodetect_endpoint(api, expected_params=len(INPUT_ORDER))
print(f"\nUsing endpoint: {target_endpoint}")

# 9) Resolve song input.
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


def _build_positional(song):
    """Merge GEN_DEFAULTS + UI_DEFAULTS + song (per-field overrides) and
    produce the positional list in INPUT_ORDER (matches the wiring)."""
    merged = dict(UI_DEFAULTS)
    # Apply GEN_DEFAULTS first (API-style names, mapped through SONG_ALIASES)
    for k, v in GEN_DEFAULTS.items():
        merged[SONG_ALIASES.get(k, k)] = v
    # Then per-song overrides
    for k, v in song.items():
        if k in ("title",):
            continue
        merged[SONG_ALIASES.get(k, k)] = v
    return [merged[name] for name in INPUT_ORDER]


# 10) Run generation, one song at a time, on the same shared init.
os.makedirs(LOCAL_OUT, exist_ok=True)
manifest = []
for i, song in enumerate(songs, 1):
    title = (song.get("title") or song.get("prompt") or song.get("caption") or f"song{i}")[:80]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\n[gen {i}/{len(songs)}] {title!r}")
    positional = _build_positional(song)
    try:
        result = client.predict(*positional, api_name=target_endpoint)
    except Exception as exc:
        # Gradio sometimes returns the chain index via fn_index instead;
        # try a couple of fallbacks.
        print(f"  predict via api_name failed: {exc}")
        fallback_indices = [1, 2, 0]
        result = None
        for idx in fallback_indices:
            try:
                result = client.predict(*positional, fn_index=idx)
                print(f"  fallback succeeded via fn_index={idx}")
                break
            except Exception as e2:
                print(f"  fn_index={idx} failed: {e2}")
        if result is None:
            raise

    # The first 8 outputs are generated_audio_1..8; with batch_size=1 only the
    # first is populated. Each value is the local server file path.
    audio_outputs = result[:8] if isinstance(result, (list, tuple)) else [result]
    saved = []
    for ai, ao in enumerate(audio_outputs, 1):
        # Gradio_client commonly returns a dict like {"path": "...", "url": "..."}
        path = ao if isinstance(ao, str) else (ao or {}).get("path") if isinstance(ao, dict) else None
        if not path:
            continue
        if not os.path.exists(path):
            print(f"  [{ai}] gradio reported path {path} but it doesn't exist")
            continue
        dst = os.path.join(LOCAL_OUT, f"{stamp}_{i:03d}_{ai}{pathlib.Path(path).suffix}")
        shutil.copy2(path, dst)
        print(f"   -> {dst} ({os.path.getsize(dst)//1024} KB)")
        saved.append(dst)
    manifest.append({"index": i, "title": title, "paths": saved, "success": bool(saved)})

# Manifest
with open(os.path.join(LOCAL_OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nmanifest -> {LOCAL_OUT}/manifest.json")

# 11) Optional Google Drive upload (Shared Drive folder required for SA).
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
    cell_source = (
        NOTEBOOK_CELL_TEMPLATE
        .replace("__INPUT_ORDER__", repr(INPUT_ORDER))
        .replace("__UI_DEFAULTS__", repr(UI_DEFAULTS))
        .replace("__SONG_ALIASES__", repr(SONG_ALIASES))
    )
    nb = {
        "cells": [{
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": cell_source,
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

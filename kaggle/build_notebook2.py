"""Generate the SECOND ACE-Step single-cell Kaggle notebook (.ipynb).

Same pipeline as build_notebook.py -- this builder imports that module's
NOTEBOOK_CELL_TEMPLATE and patches exactly TWO lines (see PATCHES):

    INPUT_JSON_PATH:  kaggle/songs.json -> kaggle/songs2.json
    MAKE_VIDEO:       False -> True

so this notebook carries an independent song stream that ALSO renders a music
video on the Kaggle box itself (Pixabay stock footage + burned lyrics + the
generated song; see the MAKE_VIDEO section in build_notebook.py) and uploads
the MP4 to Drive alongside the MP3. The song spec lands in this repo as a
PAT-authored Contents-API commit of kaggle/songs2.json (from the daily routine
chain), which fires .github/workflows/kaggle-trigger-2.yml, which pushes this
notebook to its own Kaggle kernel (allinone2015/acestep-daily-2 -- see
notebook2/kernel-metadata.json).

Everything else (secrets placeholders, stability env, REST API flow, scoring,
Drive upload) is inherited verbatim from build_notebook.py, so a fix there
automatically lands here on the next build.

Run:  python kaggle/build_notebook2.py
Output: kaggle/notebook2/acestep_kaggle_2xT4_daily2.ipynb
"""
import os
import sys

# Resolve the sibling build_notebook.py regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_notebook as base

# The only lines that differ from notebook #1 (comment columns kept aligned):
# a different input JSON, and the video stage switched on.
PATCHES = [
    ('INPUT_JSON_PATH   = f"{WORK_DIR}/kaggle/songs.json"        # committed daily by the Claude routine',
     'INPUT_JSON_PATH   = f"{WORK_DIR}/kaggle/songs2.json"       # stream #2 song spec'),
    ('MAKE_VIDEO      = False                  # patched to True by build_notebook2.py',
     'MAKE_VIDEO      = True                   # stream #2: render music videos'),
]


def build_notebook2(out_path: str) -> None:
    template = base.NOTEBOOK_CELL_TEMPLATE
    patched = template
    for old, new in PATCHES:
        if patched.count(old) != 1:
            raise SystemExit(
                f"build_notebook.py's template changed -- this patch line no longer "
                f"matches exactly once: {old!r}. Update PATCHES in build_notebook2.py."
            )
        patched = patched.replace(old, new)
    # Temporarily swap the module-level template so base.build_notebook()
    # (which reads the global at call time) emits the patched cell, then
    # restore it. Keeps a single source of truth for the notebook body.
    base.NOTEBOOK_CELL_TEMPLATE = patched
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base.build_notebook(out_path)
    finally:
        base.NOTEBOOK_CELL_TEMPLATE = template


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "notebook2",
        "acestep_kaggle_2xT4_daily2.ipynb",
    )
    build_notebook2(out)

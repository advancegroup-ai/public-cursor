# public-cursor

Workspace repository for Cursor Cloud Agent tasks.

This repo is a **minimal sandbox wrapper** used by [Vibe Track](https://github.com/simontt88/vibe-track-open) to run Cursor Cloud Agents programmatically. It intentionally does **not** include the actual CV/forgery code.

## How to work on forgery detection in Cursor

The document-forgery mission you described (e.g. `projects/forgery_cases`, CLIP background embedding ONNX, ArcFace face vectors, clustering + weekly scale-out) lives in the **real code/data repo**, not here.

To make this workspace actionable for implementation and PRs, you have two options:

### Option A (recommended): Open the real repo in this workspace

- Clone `vibe-track-open` into `/workspace` (or open it directly in Cursor).
- Run the agent against that repository so it can read existing scripts and commit code changes where they belong.

### Option B: Add `vibe-track-open` as a git submodule

If you need this wrapper repo to stay minimal, add the real repo as a submodule so Cursor can still commit PRs against it:

```bash
cd /workspace
git submodule add https://github.com/simontt88/vibe-track-open.git vibe-track-open
git submodule update --init --recursive
```

Then work under `vibe-track-open/projects/forgery_cases/` and commit changes in the submodule repo as needed.

## Why this is necessary

This `/workspace` repository currently only contains `README.md` and `.gitignore`, and there are **no references or files** for:

- `projects/forgery_cases/`
- `download_images.py`, `analyze_forgery.py`, `combo_scoring.py`
- CLIP ONNX background embedding code
- ArcFace vectorization code

So, without checking out the real repository (or adding it as a submodule), there is nothing concrete to implement or test here.

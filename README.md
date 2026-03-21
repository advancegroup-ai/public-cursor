# public-cursor

This repository is a **minimal sandbox wrapper** used to run Cursor Cloud Agents.

It **does not contain** the document-forgery detection code/data (e.g. `projects/forgery_cases/`) referenced in the mission notes. That work lives in the main Vibe Track repo:

- `simontt88/vibe-track-open` (GitHub): `https://github.com/simontt88/vibe-track-open`

## How to work on forgery detection from Cursor

To implement and test the forgery pipeline (Bybit bypass cases, embeddings, clustering), you need the real code checked out in the workspace.

### Option A (recommended): open/clone the real repo as the workspace

Clone and open `vibe-track-open` directly in Cursor, then create your feature branch there. This is the simplest route for making code changes, running scripts, and creating PRs.

### Option B: add `vibe-track-open` as a submodule here

If you must keep this wrapper repo as the workspace root, you can add the real repo as a submodule, then work inside it:

```bash
git submodule add https://github.com/simontt88/vibe-track-open.git vibe-track-open
git submodule update --init --recursive
```

Then you should run/modify code inside `./vibe-track-open/` (and open PRs from that repo).

## Why this is required

The forgery-detection work references:

- `projects/forgery_cases/` scripts (analysis, embedding extraction, clustering)
- shared model weights (e.g. background CLIP ONNX) and NAS paths

None of those assets exist in this wrapper repository, so there is nothing meaningful to implement or validate here until the real repo is present.

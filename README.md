# public-cursor

Workspace repository for Cursor Cloud Agent API tasks.

This repo is used as a sandbox by [Vibe Track](https://github.com/simontt88/vibe-track-open) to run Cursor Cloud Agents programmatically.

## What this repo is (and is not)

- **This repo is a thin “runner” workspace** for Cursor Cloud Agents.
- **It does not contain** the actual forgery analysis / embedding pipeline (e.g. `projects/forgery_cases/`), datasets, or model weights.

If your task involves document forgery detection work (embeddings, clustering, case analysis), you almost certainly want the real codebase:

- **Primary repo**: `simontt88/vibe-track-open`

## How to work on the forgery pipeline

Choose one of these approaches so the agent can implement/test the pipeline in-repo:

### Option A (recommended): open/clone the real repo in the workspace

Run the Cloud Agent with the **`vibe-track-open`** repository checked out as the workspace root. That’s where:

- `projects/forgery_cases/` scripts live (downloads, embeddings, reports)
- references like `cv-tdd-core/.../face_background_processor.py` can be wired in cleanly

### Option B: add `vibe-track-open` as a submodule

If you must keep this repo as the root, add the real repo as a submodule and point work there:

```bash
git submodule add https://github.com/simontt88/vibe-track-open.git vibe-track-open
git submodule update --init --recursive
```

Then implement and run tasks under `vibe-track-open/projects/forgery_cases/`.

## Notes on environment constraints

Cursor Cloud Agent VMs **do not automatically have access** to your internal NAS paths or private “backend tool” endpoints unless they’re explicitly provided in the environment. If you need the agent to:

- read `/mnt/nas/...` paths
- call internal endpoints for `shell/fs/python/deliver`

…make sure those resources are reachable from the agent runtime (network access + credentials/secrets).

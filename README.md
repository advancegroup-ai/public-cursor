# public-cursor

Workspace repository for Cursor Cloud Agent API tasks.

This repo is used as a sandbox by [Vibe Track](https://github.com/simontt88/vibe-track-open) to run Cursor Cloud Agents programmatically.

## Forgery detection utilities

This repository now includes a minimal Python package under `src/forgery_detection` for
forgery-analysis primitives:

- NPZ vector-store load/save (`NpzVectorStore`)
- Normalized ID overlap join (`join_on_normalized_ids`)
- Cosine-threshold connected-components clustering (`cluster_by_cosine_threshold`)
- Face bbox zero-mask helper (`mask_bbox_zero_rgb`)

### Local validation

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

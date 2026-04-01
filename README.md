# public-cursor

Workspace repository for Cursor Cloud Agent API tasks.

This repo is used as a sandbox by [Vibe Track](https://github.com/simontt88/vibe-track-open) to run Cursor Cloud Agents programmatically.

## Forgery document + face embedding helpers

This repo now includes a minimal Python package in `src/forgery_document_face_embeddings` to support:

- doc-front background embedding with optional face masking (CLIP ONNX wrapper),
- ArcFace face embedding wrapper (InsightFace),
- normalized ID intersection diagnostics,
- cosine-threshold clustering for analysis.

### Install

```bash
pip install -e .[dev]
```

Optional model/runtime extras:

```bash
pip install -e .[onnx,cv,face]
```

### Run tests

```bash
pytest -q
```

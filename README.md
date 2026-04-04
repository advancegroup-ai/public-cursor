# public-cursor

This repo is a lightweight **Cursor Cloud Agent sandbox**. It currently does **not** contain the internal `forgery_cases/` pipeline referenced in your “Atlas” runbook.

What I *can* do here is keep a **portable prototype** of the two embedding primitives you described, so it’s easy to transplant into the real `vibe-track-open` / NAS environment once that codebase is available in this workspace.

## Prototype embeddings added here

- **Doc-front background embedding (face excluded)**: `src/atlas_forgery/embeddings/doc_front_bg_clip_onnx.py`
  - loads an ONNX CLIP-style embedder
  - detects a face (best-effort via `insightface`) and **zeros out** that bbox before embedding
  - returns a **512-dim** L2-normalized vector
- **Liveness face embedding (ArcFace)**: `src/atlas_forgery/embeddings/arcface_insightface.py`
  - uses `insightface` “buffalo_l” as a practical stand-in for your guardian ArcFace pipeline
  - returns a **512-dim** L2-normalized vector

## Quick local usage

Install:

```bash
pip install -e .
pip install -e ".[face]"   # optional: face detect + arcface
```

Run:

```bash
python3 scripts/embed_single.py \
  --doc-front /path/to/doc_front.jpg \
  --liveness /path/to/liveness.jpg \
  --clip-onnx /path/to/background_embedding_CLIP-global-v1.4.2.onnx
```

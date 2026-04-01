# public-cursor

Workspace repository for Cursor Cloud Agent API tasks.

This repo is used as a sandbox by [Vibe Track](https://github.com/simontt88/vibe-track-open) to run Cursor Cloud Agents programmatically.

## atlas-forgery

This repo currently contains a small Python package, `atlas-forgery`, intended to support
document forgery investigations via:

- Face-masked document embeddings (baseline + optional ONNX backend)
- Face embeddings (deterministic baseline; ArcFace backend can be added/used as needed)
- Cosine-threshold clustering and simple JSON reporting

### Dev setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m ruff check .
python3 -m pytest -q
```

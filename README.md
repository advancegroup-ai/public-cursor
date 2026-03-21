# public-cursor

Workspace repository for Cursor Cloud Agent API tasks.

This repo is used as a sandbox by [Vibe Track](https://github.com/simontt88/vibe-track-open) to run Cursor Cloud Agents programmatically.

## atlas-forgery (portable forgery primitives)

This repo intentionally does **not** include internal NAS paths, ODPS access, or private model weights.
To still make progress on forgery-detection experiments, it contains a small Python package with:

- `DocFrontBackgroundClipOnnxEmbedder`: run a CLIP-style ONNX encoder with optional face-region zero-masking
- `ArcFaceInsightFaceEmbedder` (optional extra): 512D face embeddings via InsightFace
- `atlas-cluster-join`: join background + face embeddings by ID and run DBSCAN clustering

Install:

```bash
python -m pip install -e ".[face]"
```

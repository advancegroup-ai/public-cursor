# atlas-doc-forgery

Reference pipeline for document forgery analysis with:

- `doc_front` background embeddings with face exclusion (mask then embed),
- liveness face embeddings (ArcFace via InsightFace),
- overlap diagnostics between background and face embedding artifacts.

## Install

```bash
python3 -m pip install --user -r requirements.txt
```

## CLI

Run via module mode:

```bash
PYTHONPATH=src python3 -m atlas_doc_forgery.cli --help
```

### 1) Embed doc_front background (face masked)

```bash
PYTHONPATH=src python3 -m atlas_doc_forgery.cli embed-doc-front-bg \
  --image-dir /path/to/doc_front_images \
  --out-npz /path/to/out/doc_front_bg_embeddings.npz \
  --clip-onnx /raid/simon/modelcenter/background_embedding_CLIP-global-v1.4.2/onnx/background_embedding_CLIP-global-v1.4.2.onnx \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --out-failures-json /path/to/out/doc_front_failures.json
```

Output NPZ schema:

- `embeddings`: float32 matrix, shape `[N, 512]` (model dependent),
- `ids`: normalized `signature_id` list aligned to rows.

### 2) Embed liveness faces (ArcFace)

```bash
PYTHONPATH=src python3 -m atlas_doc_forgery.cli embed-face \
  --image-dir /path/to/liveness_images \
  --out-npz /path/to/out/liveness_face_embeddings.npz \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --out-failures-json /path/to/out/face_failures.json
```

### 3) Evaluate signature overlap (join health check)

```bash
PYTHONPATH=src python3 -m atlas_doc_forgery.cli eval-join \
  --bg-npz /path/to/out/doc_front_bg_embeddings.npz \
  --face-npz /path/to/out/liveness_face_embeddings.npz \
  --out-join-csv /path/to/out/intersection.csv
```

This step directly surfaces the common failure mode where `intersection_n == 0` due to mismatched
key normalization or filename-derived IDs.

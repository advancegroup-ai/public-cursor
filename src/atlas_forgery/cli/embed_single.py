from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from atlas_forgery.embeddings.doc_front_bg_clip_onnx import DocFrontBackgroundEmbedderOnnx


def _parse_bbox(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not s:
        return None
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed a single doc-front image with optional face masking.")
    ap.add_argument("--onnx", required=True, help="Path to background CLIP-style ONNX model")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--face-bbox", default=None, help="Face bbox x1,y1,x2,y2 to zero-mask before embedding")
    ap.add_argument("--out", default=None, help="Optional output .npy path for embedding")
    args = ap.parse_args()

    embedder = DocFrontBackgroundEmbedderOnnx(args.onnx)
    face_bbox = _parse_bbox(args.face_bbox)
    emb = embedder.embed_path(args.image, face_bbox_xyxy=face_bbox)

    payload = {"dim": int(emb.shape[0]), "norm": float(np.linalg.norm(emb)), "head": emb[:8].tolist()}
    print(json.dumps(payload, ensure_ascii=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb.astype(np.float32))


if __name__ == "__main__":
    main()


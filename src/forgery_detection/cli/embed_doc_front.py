from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder
from forgery_detection.embeddings.image_utils import parse_bbox


def _build_session(onnx_path: str):
    import onnxruntime as ort  # optional dependency

    return ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Compute doc_front background embedding (optionally mask face).")
    p.add_argument("--image", required=True, help="Path to doc_front image")
    p.add_argument("--onnx", required=True, help="Path to CLIP background embedding ONNX model")
    p.add_argument("--input-name", default="input", help="ONNX input name (default: input)")
    p.add_argument("--face-bbox", default=None, help="Optional bbox: x1,y1,x2,y2 (pixels)")
    p.add_argument("--no-l2", action="store_true", help="Disable L2 normalization")
    p.add_argument("--out", default=None, help="Optional output .npy path")
    args = p.parse_args()

    bbox = None if args.face_bbox is None else parse_bbox(args.face_bbox)
    sess = _build_session(args.onnx)
    embedder = DocFrontBackgroundEmbedder(
        session=sess,
        input_name=args.input_name,
        l2_normalize=not args.no_l2,
    )
    vec = embedder.embed(args.image, face_bbox=bbox)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vec)
    else:
        print(json.dumps({"dim": int(vec.shape[0]), "norm": float(np.linalg.norm(vec))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


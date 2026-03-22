from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import (
    DocFrontBackgroundEmbedder,
    load_onnx_session,
)


def _parse_bbox(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'x1,y1,x2,y2'")
    try:
        x1, y1, x2, y2 = (int(float(p)) for p in parts)
    except Exception as e:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid bbox: {e}") from e
    return (x1, y1, x2, y2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed doc_front background with face masking.")
    ap.add_argument("--model", required=True, help="Path to CLIP background ONNX model")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--providers", default="", help="Comma-separated ORT providers (optional)")
    ap.add_argument("--face-bbox", default=None, type=_parse_bbox, help="Face bbox xyxy to mask")
    ap.add_argument("--out", default=None, help="Optional output .npy path")
    ap.add_argument("--no-norm", action="store_true", help="Disable L2 normalization")
    args = ap.parse_args()

    providers = [p for p in (x.strip() for x in args.providers.split(",")) if p] or None
    sess = load_onnx_session(args.model, providers=providers)
    embedder = DocFrontBackgroundEmbedder(sess, l2_normalize=not args.no_norm)

    img = Image.open(args.image)
    emb = embedder.embed_pil(img, face_bbox_xyxy=args.face_bbox)

    payload = {
        "dim": int(emb.shape[0]),
        "l2_norm": float(np.linalg.norm(emb)),
        "min": float(emb.min()),
        "max": float(emb.max()),
    }
    print(json.dumps(payload, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

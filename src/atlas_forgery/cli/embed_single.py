from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from atlas_forgery.embeddings.doc_front_bg_clip_onnx import DocFrontBackgroundEmbedderONNX, MaskBox


def _parse_mask_boxes(values: list[str]) -> list[MaskBox]:
    out: list[MaskBox] = []
    for v in values:
        parts = [int(x) for x in v.split(",")]
        if len(parts) != 4:
            raise ValueError(f"mask box must be x0,y0,x1,y1; got {v}")
        out.append(MaskBox(*parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed a single image with ONNX CLIP-style model.")
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--mask-box", action="append", default=[], help="Mask region x0,y0,x1,y1 (can repeat)")
    ap.add_argument("--out", default="", help="Output .npy path (optional)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    embedder = DocFrontBackgroundEmbedderONNX(args.onnx)
    mask_boxes = _parse_mask_boxes(args.mask_box)
    vec = embedder.embed_bgr(img, mask_boxes=mask_boxes)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, vec)

    print(json.dumps({"dim": int(vec.shape[0]), "norm": float(np.linalg.norm(vec))}))


if __name__ == "__main__":
    main()

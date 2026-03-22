from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder


def _parse_bbox_xyxy(s: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected bbox format: x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(float(p)) for p in parts)
    return x1, y1, x2, y2


def _iter_floats(vec: np.ndarray) -> Iterable[float]:
    for x in vec.reshape(-1).tolist():
        yield float(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute doc_front background embedding (face optional).")
    ap.add_argument("--image", required=True, help="Path to doc_front image")
    ap.add_argument("--onnx", required=True, help="Path to CLIP background embedding ONNX model")
    ap.add_argument("--input-name", default="image", help="ONNX input tensor name (default: image)")
    ap.add_argument("--output-name", default=None, help="ONNX output tensor name (default: first output)")
    ap.add_argument("--face-bbox", default=None, help="Optional face bbox x1,y1,x2,y2 (pixels)")
    ap.add_argument("--no-l2", action="store_true", help="Disable L2 normalization")
    ap.add_argument("--format", choices=["json", "npy"], default="json", help="Output format")
    ap.add_argument("--out", default="-", help="Output path or '-' for stdout")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "onnxruntime is required for this CLI. Install with: pip install onnxruntime"
        ) from e

    model_path = str(Path(args.onnx))
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    embedder = DocFrontBackgroundEmbedder(
        session=sess,
        input_name=args.input_name,
        output_name=args.output_name,
        l2_normalize=not args.no_l2,
    )

    img = Image.open(args.image)
    bbox = _parse_bbox_xyxy(args.face_bbox) if args.face_bbox else None
    vec = embedder.embed(img, face_bbox_xyxy=bbox)

    if args.format == "npy":
        if args.out == "-":
            raise SystemExit("--format npy requires --out to be a file path")
        np.save(args.out, vec)
        return

    payload = {"dim": int(vec.shape[0]), "embedding": list(_iter_floats(vec))}
    text = json.dumps(payload, ensure_ascii=False)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()


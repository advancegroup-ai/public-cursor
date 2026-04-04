from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder


def _parse_bbox(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not s:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x1,y1,x2,y2'")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed doc_front background with optional face masking.")
    ap.add_argument("--onnx", required=True, help="Path to CLIP background embedding ONNX model.")
    ap.add_argument("--image", required=True, help="Path to doc_front image.")
    ap.add_argument("--out", required=True, help="Output .npy path.")
    ap.add_argument("--input-name", default="input", help="ONNX input tensor name.")
    ap.add_argument("--output-name", default=None, help="Optional ONNX output name.")
    ap.add_argument("--face-bbox", default=None, help="Optional face bbox 'x1,y1,x2,y2' to mask (zero).")
    ap.add_argument("--no-l2", action="store_true", help="Disable L2 normalization.")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"onnxruntime is required for this CLI: {e}")

    sess = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])
    embedder = DocFrontBackgroundEmbedder(
        session=sess,
        input_name=args.input_name,
        output_name=args.output_name,
        l2_normalize=not args.no_l2,
    )
    img = Image.open(args.image)
    bbox = _parse_bbox(args.face_bbox)
    vec = embedder.embed_pil(img, face_bbox=bbox)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, vec.astype(np.float32))


if __name__ == "__main__":  # pragma: no cover
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder


def _parse_bbox(s: Optional[str]) -> Optional[Sequence[int]]:
  if not s:
    return None
  parts = [p.strip() for p in s.split(",")]
  if len(parts) != 4:
    raise ValueError("bbox must be x1,y1,x2,y2")
  return [int(p) for p in parts]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True, help="Path to ONNX model")
  ap.add_argument("--image", required=True, help="Path to doc_front image")
  ap.add_argument("--face-bbox", default=None, help="Optional x1,y1,x2,y2 to zero-mask")
  ap.add_argument("--out", default=None, help="Output .npy path (default: stdout json)")
  args = ap.parse_args()

  import onnxruntime as ort  # lazy import

  sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
  input_name = sess.get_inputs()[0].name

  img = np.asarray(Image.open(args.image).convert("RGB"))
  bbox = _parse_bbox(args.face_bbox)
  embedder = DocFrontBackgroundEmbedder(session=sess, input_name=input_name)
  vec = embedder.embed_rgb_uint8(img, face_bbox_xyxy=bbox)

  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vec)
  else:
    print(
      json.dumps(
        {"dim": int(vec.shape[0]), "l2": float(np.linalg.norm(vec)), "vector_head": vec[:8].tolist()}
      )
    )
 

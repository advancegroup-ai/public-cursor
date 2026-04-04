from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from forgery_detection.embeddings.arcface import ArcFaceEmbedder, build_insightface_arcface_backend


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--image", required=True, help="Path to 112x112 aligned face image")
  ap.add_argument("--out", default=None, help="Output .npy path (default: stdout json)")
  ap.add_argument("--ctx-id", type=int, default=-1, help="insightface ctx_id (-1 CPU, >=0 GPU)")
  args = ap.parse_args()

  aligned = np.asarray(Image.open(args.image))
  backend = build_insightface_arcface_backend(ctx_id=args.ctx_id)
  embedder = ArcFaceEmbedder(backend=backend)
  vec = embedder.embed_aligned_face(aligned)

  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vec)
  else:
    print(
      json.dumps(
        {"dim": int(vec.shape[0]), "l2": float(np.linalg.norm(vec)), "vector_head": vec[:8].tolist()}
      )
    )
 

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from atlas_forgery.face_embed import DeterministicFaceEmbedder
from atlas_forgery.vector_store import VectorStore, save_npz


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="Aligned 112x112 images (BGR or RGB)")
    ap.add_argument("--ids", nargs="+", required=True, help="List of ids (same length as images)")
    ap.add_argument("--out", required=True, help="Output .npz path")
    args = ap.parse_args(argv)

    if len(args.images) != len(args.ids):
        raise SystemExit("images and ids must have same length")

    embedder = DeterministicFaceEmbedder()
    vecs: list[np.ndarray] = []
    for img_path in args.images:
        im = Image.open(Path(img_path)).convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8)
        if rgb.shape[:2] != (112, 112):
            raise SystemExit(f"Expected 112x112 image; got {rgb.shape}")
        bgr = rgb[..., ::-1].copy()
        vecs.append(embedder.embed_aligned_112(bgr))

    vectors = np.stack(vecs, axis=0).astype(np.float32)
    store = VectorStore(ids=[str(x) for x in args.ids], vectors=vectors, meta={"type": "face"})
    save_npz(args.out, store)


if __name__ == "__main__":
    main()


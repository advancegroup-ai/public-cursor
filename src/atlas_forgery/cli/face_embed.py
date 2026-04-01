from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..embedders import DeterministicFaceEmbedder
from ..vectors import VectorStore, save_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--faces",
        nargs="+",
        required=True,
        help="Paths to aligned 112x112 face crops.",
    )
    ap.add_argument("--ids", nargs="+", required=True, help="IDs aligned with --faces.")
    ap.add_argument("--out", required=True, help="Output .npz")
    args = ap.parse_args()

    if len(args.faces) != len(args.ids):
        raise SystemExit("--faces and --ids length mismatch")

    emb = DeterministicFaceEmbedder(dim=512)
    vecs: list[np.ndarray] = []
    for p in args.faces:
        # Support RGB inputs; convert to BGR for interface.
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        if bgr.shape[:2] != (112, 112):
            raise SystemExit(f"Expected 112x112, got {bgr.shape} for {p}")
        vecs.append(emb.embed_aligned_bgr112(bgr))

    store = VectorStore(ids=np.asarray(args.ids, dtype=object), vectors=np.stack(vecs, axis=0))
    save_npz(Path(args.out), store)


if __name__ == "__main__":
    main()


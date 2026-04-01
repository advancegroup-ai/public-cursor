from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..embed.face import InsightFaceArcFaceEmbedder, MeanRGBFaceEmbedder
from ..io import save_npz


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", nargs="+", required=True, help="Paths to aligned 112x112 RGB faces.")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--insightface", action="store_true", help="Use insightface ArcFace embedder.")
    args = ap.parse_args()

    embedder = InsightFaceArcFaceEmbedder() if args.insightface else MeanRGBFaceEmbedder()

    ids = []
    vecs = []
    for p in map(Path, args.faces):
        img_id = p.stem
        v = embedder.embed_aligned(_load_rgb(p))
        ids.append(img_id)
        vecs.append(v)

    save_npz(
        args.out,
        np.asarray(ids, dtype=str),
        np.stack(vecs).astype(np.float32),
        meta={"kind": "face"},
    )


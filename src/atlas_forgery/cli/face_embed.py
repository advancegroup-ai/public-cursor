from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from atlas_forgery.embedding import DeterministicFaceEmbedder
from atlas_forgery.io import load_rgb
from atlas_forgery.vector_store import save_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed aligned 112x112 face crops to 512-d vectors.")
    ap.add_argument("--csv", required=True, help="CSV with columns: id,face_image_path")
    ap.add_argument("--out", required=True, help="Output .npz (ids + vectors)")
    ap.add_argument("--dim", type=int, default=512)
    args = ap.parse_args()

    embedder = DeterministicFaceEmbedder(output_dim=args.dim)
    ids: list[str] = []
    vecs: list[np.ndarray] = []

    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames or "face_image_path" not in reader.fieldnames:
            raise SystemExit("CSV must contain columns: id,face_image_path")
        for row in reader:
            sid = str(row["id"])
            img = load_rgb(row["face_image_path"])
            v = embedder.get_feature(img)
            ids.append(sid)
            vecs.append(v.astype(np.float32))

    vectors = np.stack(vecs, axis=0) if vecs else np.zeros((0, args.dim), dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out, ids=ids, vectors=vectors)


if __name__ == "__main__":
    main()

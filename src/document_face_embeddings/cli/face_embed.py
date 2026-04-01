from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..face_embedding import DeterministicFaceEmbedder
from ..vector_store import save_npz


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: id,face_bgr_npy")
    ap.add_argument("--out", required=True, help="Output .npz path")
    args = ap.parse_args(argv)

    embedder = DeterministicFaceEmbedder()
    ids: list[str] = []
    vecs: list[np.ndarray] = []

    with open(args.csv, "r", newline="") as f:
        r = csv.DictReader(f)
        if "id" not in r.fieldnames or "face_bgr_npy" not in r.fieldnames:
            raise SystemExit("CSV must contain columns: id,face_bgr_npy")
        for row in r:
            _id = str(row["id"])
            face = np.load(row["face_bgr_npy"])
            v = embedder.embed_face_bgr(face)
            ids.append(_id)
            vecs.append(v.astype(np.float32))

    vectors = np.stack(vecs, axis=0) if vecs else np.zeros((0, embedder.dim), np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out, ids=ids, vectors=vectors)


if __name__ == "__main__":  # pragma: no cover
    main()


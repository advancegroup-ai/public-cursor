from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from atlas_forgery.embedding import DeterministicImageEmbedder, FaceMaskedDocFrontEmbedder
from atlas_forgery.io import BBoxXYXY, load_rgb
from atlas_forgery.vector_store import save_npz


def _parse_bbox(s: str) -> BBoxXYXY | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 ints, got: {s!r}")
    x1, y1, x2, y2 = (int(p) for p in parts)
    return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed doc_front images with optional face masking.")
    ap.add_argument("--csv", required=True, help="CSV with columns: id,image_path[,face_bbox_xyxy]")
    ap.add_argument("--out", required=True, help="Output .npz (ids + vectors)")
    ap.add_argument("--dim", type=int, default=512)
    args = ap.parse_args()

    base = DeterministicImageEmbedder(output_dim=args.dim)
    embedder = FaceMaskedDocFrontEmbedder(base_embedder=base)

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames or "image_path" not in reader.fieldnames:
            raise SystemExit("CSV must contain columns: id,image_path[,face_bbox_xyxy]")
        for row in reader:
            sid = str(row["id"])
            img = load_rgb(row["image_path"])
            bbox = _parse_bbox(row.get("face_bbox_xyxy", "") or "")
            v = embedder.embed(img, bbox)
            ids.append(sid)
            vecs.append(v.astype(np.float32))

    vectors = np.stack(vecs, axis=0) if vecs else np.zeros((0, args.dim), dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out, ids=ids, vectors=vectors)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..embedding import MaskedEmbedder, MeanRGBEmbedder
from ..io import BBoxXYXY, load_rgb
from ..vector_store import save_npz


def _parse_bbox(s: str) -> BBoxXYXY | None:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = (int(float(p)) for p in parts)
    return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: id,image_path[,face_bbox]")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument(
        "--face-bbox-col",
        default="face_bbox",
        help="Optional bbox column: 'x1,y1,x2,y2' (default: face_bbox)",
    )
    args = ap.parse_args(argv)

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    embedder = MaskedEmbedder(base=MeanRGBEmbedder())

    with open(args.csv, "r", newline="") as f:
        r = csv.DictReader(f)
        if "id" not in r.fieldnames or "image_path" not in r.fieldnames:
            raise SystemExit("CSV must contain columns: id,image_path")
        for row in r:
            _id = str(row["id"])
            img = load_rgb(row["image_path"])
            bb = _parse_bbox(row.get(args.face_bbox_col, "")) if args.face_bbox_col else None
            v = embedder.embed(img, face_bbox=bb)
            ids.append(_id)
            vecs.append(v.astype(np.float32))

    out = Path(args.out)
    vectors = np.stack(vecs, axis=0) if vecs else np.zeros((0, embedder.dim), np.float32)
    save_npz(out, ids=ids, vectors=vectors)


if __name__ == "__main__":  # pragma: no cover
    main()


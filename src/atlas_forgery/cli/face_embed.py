from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from atlas_forgery.face_embedding import (
    DeterministicFaceEmbedder,
    InsightFaceArcFaceEmbedder,
    load_rgb,
)
from atlas_forgery.vector_store import save_npz


def _iter_image_rows(input_path: Path, id_column: str, image_column: str):
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get(id_column)
            img = row.get(image_column)
            if sid and img:
                yield sid, img


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create 512-d face embeddings and save to NPZ.")
    p.add_argument("--input-csv", type=Path, required=True, help="CSV with signature/image mapping.")
    p.add_argument("--image-root", type=Path, default=Path("."), help="Base dir for image paths.")
    p.add_argument("--output", type=Path, required=True, help="Output NPZ path.")
    p.add_argument("--id-column", default="signature_id", help="CSV column for id.")
    p.add_argument("--image-column", default="image_path", help="CSV column for image path.")
    p.add_argument(
        "--backend",
        choices=["deterministic", "insightface"],
        default="deterministic",
        help="Embedding backend.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    rows = list(_iter_image_rows(args.input_csv, args.id_column, args.image_column))
    if not rows:
        raise SystemExit("No valid rows found in input CSV.")

    if args.backend == "insightface":
        embedder = InsightFaceArcFaceEmbedder()
    else:
        embedder = DeterministicFaceEmbedder()

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    failures: list[tuple[str, str]] = []

    for sid, rel_path in rows:
        image_path = args.image_root / rel_path
        try:
            image = load_rgb(image_path)
            vector = embedder.embed(image)
            ids.append(sid)
            vectors.append(vector)
        except Exception as exc:  # pragma: no cover
            failures.append((sid, str(exc)))

    if not vectors:
        raise SystemExit("All rows failed during embedding.")

    ids_arr = np.asarray(ids, dtype="<U128")
    vec_arr = np.vstack(vectors).astype(np.float32, copy=False)
    save_npz(args.output, ids_arr, vec_arr)

    print(f"embedded={len(vectors)} failed={len(failures)} output={args.output}")
    if failures:
        print("sample_failures:")
        for sid, msg in failures[:5]:
            print(f"  {sid}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

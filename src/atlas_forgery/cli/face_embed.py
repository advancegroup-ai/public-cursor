from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..embed.face import DeterministicFaceEmbedder, InsightFaceArcFaceEmbedder
from ..npz_store import NpzVectorStore


def main() -> None:
    p = argparse.ArgumentParser(description="Embed aligned face crops (112x112) to 512-d vectors.")
    p.add_argument("--images-dir", required=True, help="Directory of face images (recursively).")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument(
        "--mode",
        choices=["deterministic", "insightface-arcface"],
        default="deterministic",
        help="Embedding backend.",
    )
    p.add_argument("--glob", default="*.jpg", help="Glob to match within images-dir.")
    p.add_argument(
        "--id-from",
        choices=["stem", "relpath"],
        default="stem",
        help="How to derive ids: filename stem or relative path.",
    )
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    paths = sorted(images_dir.rglob(args.glob))
    if not paths:
        raise SystemExit(f"No images found under {images_dir} with glob={args.glob!r}")

    if args.mode == "insightface-arcface":
        embedder = InsightFaceArcFaceEmbedder()
    else:
        embedder = DeterministicFaceEmbedder()

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for path in paths:
        sid = str(path.relative_to(images_dir)) if args.id_from == "relpath" else path.stem
        ids.append(sid)
        vecs.append(embedder.embed_path(path))

    store = NpzVectorStore(ids=ids, vectors=np.stack(vecs, axis=0))
    store.save(args.out)
    print(f"Wrote {len(store.ids)} vectors dim={store.dim} -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from document_face_embeddings.embedding import MaskedEmbedder, MeanRGBEmbedder
from document_face_embeddings.io import BBoxXYXY, load_image_rgb
from document_face_embeddings.vector_store import VectorStore, save_npz


def _load_face_bboxes_json(path: str | Path) -> dict[str, BBoxXYXY]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, BBoxXYXY] = {}
    # Expected: { "id": [x1,y1,x2,y2], ... }
    for k, v in raw.items():
        if not (isinstance(v, list) and len(v) == 4):
            continue
        out[str(k)] = BBoxXYXY(int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed doc images with optional face masking.")
    ap.add_argument("--images-dir", required=True, help="Directory containing images.")
    ap.add_argument("--glob", default="*.jpg", help="Glob pattern (default: *.jpg).")
    ap.add_argument("--output-npz", required=True, help="Output .npz with ids + vectors.")
    ap.add_argument(
        "--face-bboxes-json",
        default=None,
        help='Optional JSON mapping id -> [x1,y1,x2,y2] to zero out before embedding.',
    )
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    paths = sorted(images_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No images matched {args.glob} in {images_dir}")

    bboxes: dict[str, BBoxXYXY] = {}
    if args.face_bboxes_json:
        bboxes = _load_face_bboxes_json(args.face_bboxes_json)

    embedder = MaskedEmbedder(base=MeanRGBEmbedder())
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for p in paths:
        id_ = p.stem
        rgb = load_image_rgb(p)
        vec = embedder.embed(rgb, face_bbox=bboxes.get(id_))
        ids.append(id_)
        vecs.append(vec.astype(np.float32))

    mat = np.stack(vecs, axis=0)
    save_npz(args.output_npz, VectorStore(ids=ids, vectors=mat))
    print(f"embedded={len(ids)} dim={mat.shape[1]} -> {args.output_npz}")


if __name__ == "__main__":
    main()


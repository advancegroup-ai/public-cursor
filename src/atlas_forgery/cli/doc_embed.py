from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atlas_forgery.doc_embed import MaskedDocEmbedder, MeanRGBEmbedder, load_rgb
from atlas_forgery.vector_store import VectorStore, save_npz


def _parse_bbox(s: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'x0,y0,x1,y1'")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="List of image paths")
    ap.add_argument("--ids", nargs="+", required=True, help="List of ids (same length as images)")
    ap.add_argument("--face-bboxes-json", default=None, help="JSON dict id -> [x0,y0,x1,y1]")
    ap.add_argument("--out", required=True, help="Output .npz path")
    args = ap.parse_args(argv)

    if len(args.images) != len(args.ids):
        raise SystemExit("images and ids must have same length")

    bbox_map: dict[str, tuple[int, int, int, int]] = {}
    if args.face_bboxes_json:
        bbox_raw = json.loads(Path(args.face_bboxes_json).read_text(encoding="utf-8"))
        for k, v in bbox_raw.items():
            bbox_map[str(k)] = tuple(int(x) for x in v)  # type: ignore[assignment]

    embedder = MaskedDocEmbedder(embedder=MeanRGBEmbedder())
    vecs: list[np.ndarray] = []
    for img_path, sid in zip(args.images, args.ids, strict=True):
        rgb = load_rgb(img_path)
        bbox = bbox_map.get(str(sid))
        vecs.append(embedder.embed(rgb, bbox))

    vectors = np.stack(vecs, axis=0).astype(np.float32)
    store = VectorStore(ids=[str(x) for x in args.ids], vectors=vectors, meta={"type": "doc"})
    save_npz(args.out, store)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..embed.doc import MaskedDocEmbedder, MeanRGBEmbedder, OnnxClipEmbedder
from ..mask import BBoxXYXY
from ..npz_store import NpzVectorStore


def _parse_bbox(s: str) -> BBoxXYXY:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x1,y1,x2,y2'")
    return BBoxXYXY(*parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Embed doc_front images with optional face-bbox masking."
    )
    p.add_argument("--images-dir", required=True, help="Directory of doc images (recursively).")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument(
        "--mode",
        choices=["meanrgb", "onnx-clip"],
        default="meanrgb",
        help="Embedding backend.",
    )
    p.add_argument("--onnx-path", default=None, help="ONNX model path (when mode=onnx-clip).")
    p.add_argument("--mask-bbox", default=None, help="Face bbox to zero: x1,y1,x2,y2")
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

    if args.mode == "onnx-clip":
        if not args.onnx_path:
            raise SystemExit("--onnx-path required when --mode=onnx-clip")
        base = OnnxClipEmbedder(args.onnx_path)
    else:
        base = MeanRGBEmbedder()

    bbox = _parse_bbox(args.mask_bbox) if args.mask_bbox else None
    embedder = MaskedDocEmbedder(base=base, face_bbox=bbox)

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for path in paths:
        if args.id_from == "relpath":
            sid = str(path.relative_to(images_dir))
        else:
            sid = path.stem
        ids.append(sid)
        vecs.append(embedder.embed_path(path))

    store = NpzVectorStore(ids=ids, vectors=np.stack(vecs, axis=0))
    store.save(args.out)
    print(f"Wrote {len(store.ids)} vectors dim={store.dim} -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

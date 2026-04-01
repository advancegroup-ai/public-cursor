from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..embedders import MaskedDocEmbedder, MeanRGBEmbedder, OnnxClipEmbedder
from ..masking import BBoxXYXY
from ..vectors import VectorStore, save_npz


def _parse_bbox(s: str) -> BBoxXYXY:
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected x1,y1,x2,y2")
    return BBoxXYXY(*parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="Paths to doc_front images.")
    ap.add_argument("--ids", nargs="+", required=True, help="IDs aligned with --images.")
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument(
        "--face-bboxes-json",
        default=None,
        help="Optional JSON mapping id-> [x1,y1,x2,y2]",
    )
    ap.add_argument("--onnx-model", default=None, help="Optional ONNX model path")
    args = ap.parse_args()

    if len(args.images) != len(args.ids):
        raise SystemExit("--images and --ids length mismatch")

    if args.onnx_model:
        base = OnnxClipEmbedder(model_path=Path(args.onnx_model))
    else:
        base = MeanRGBEmbedder(dim=512)
    embedder = MaskedDocEmbedder(base=base)

    bbox_by_id: dict[str, BBoxXYXY] = {}
    if args.face_bboxes_json:
        raw = json.loads(Path(args.face_bboxes_json).read_text())
        for k, v in raw.items():
            bbox_by_id[str(k)] = BBoxXYXY(*[int(x) for x in v])

    vecs: list[np.ndarray] = []
    for img_path, _id in zip(args.images, args.ids, strict=True):
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        v = embedder.embed_rgb_with_face_bbox(rgb, bbox_by_id.get(_id))
        vecs.append(v)

    store = VectorStore(ids=np.asarray(args.ids, dtype=object), vectors=np.stack(vecs, axis=0))
    save_npz(args.out, store)


if __name__ == "__main__":
    main()


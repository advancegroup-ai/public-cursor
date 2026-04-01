from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..embed.doc import MaskedDocEmbedder, MeanRGBEmbedder, OnnxClipEmbedder
from ..io import save_npz
from ..types import BBoxXYXY


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="Paths to RGB images.")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--onnx-model", default=None, help="Optional ONNX model path.")
    ap.add_argument("--face-bboxes-json", default=None, help="Optional JSON mapping id->bbox.")
    args = ap.parse_args()

    embedder = MeanRGBEmbedder() if args.onnx_model is None else OnnxClipEmbedder(args.onnx_model)
    masked = MaskedDocEmbedder(embedder=embedder)

    bbox_map = {}
    if args.face_bboxes_json:
        bbox_map = json.loads(Path(args.face_bboxes_json).read_text(encoding="utf-8"))

    ids = []
    vecs = []
    for p in map(Path, args.images):
        img_id = p.stem
        bb = None
        if img_id in bbox_map and bbox_map[img_id] is not None:
            x0, y0, x1, y1 = bbox_map[img_id]
            bb = BBoxXYXY(int(x0), int(y0), int(x1), int(y1))
        v = masked.embed(_load_rgb(p), bb)
        ids.append(img_id)
        vecs.append(v)

    ids_arr = np.asarray(ids, dtype=str)
    vecs_arr = np.stack(vecs).astype(np.float32)
    save_npz(args.out, ids_arr, vecs_arr, meta={"kind": "doc", "masked": True})


from __future__ import annotations

import argparse
import csv

import numpy as np

from ..doc_embedding import MeanRGBEmbedder, MaskedDocEmbedder, OnnxClipEmbedder
from ..image_io import load_rgb
from ..masking import BBoxXYXY
from ..vector_store import save_npz


def _parse_bbox(s: str | None) -> BBoxXYXY | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = (int(float(p)) for p in parts)
    return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed doc_front images with optional face masking.")
    ap.add_argument("--csv", required=True, help="CSV with columns: id,path[,face_bbox].")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--model", choices=["meanrgb", "onnx"], default="meanrgb")
    ap.add_argument("--onnx-path", default=None, help="ONNX model path (required if --model=onnx).")
    ap.add_argument(
        "--face-bbox-col",
        default="face_bbox",
        help="Column containing 'x1,y1,x2,y2' (optional).",
    )
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--path-col", default="path")
    args = ap.parse_args()

    if args.model == "onnx":
        if not args.onnx_path:
            raise SystemExit("--onnx-path is required when --model=onnx")
        base = OnnxClipEmbedder(args.onnx_path)
        model_name = "onnx"
    else:
        base = MeanRGBEmbedder()
        model_name = "meanrgb"

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    failures: list[tuple[str, str]] = []

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row[args.id_col])
            p = row[args.path_col]
            bbox = _parse_bbox(row.get(args.face_bbox_col))
            try:
                img = load_rgb(p).array
                emb = MaskedDocEmbedder(base=base, face_bbox=bbox).embed_rgb(img)
                ids.append(sid)
                vecs.append(emb)
            except Exception as e:
                failures.append((sid, f"{type(e).__name__}: {e}"))

    vectors = np.stack(vecs, axis=0) if vecs else np.zeros((0, 512), dtype=np.float32)
    save_npz(
        args.out,
        ids=np.asarray(ids, dtype=object),
        vectors=vectors,
        model=model_name,
        total_rows=len(ids) + len(failures),
        embedded=len(ids),
        failed=len(failures),
        failures=failures[:50],
    )


if __name__ == "__main__":
    main()


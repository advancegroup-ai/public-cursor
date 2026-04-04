from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forgery_detection.embeddings.background_clip_onnx import ClipBackgroundEmbedder
from forgery_detection.embeddings.face_masking import zero_out_bbox
from forgery_detection.io.vectors import save_vectors
from forgery_detection.types import BBoxXYXY


def _load_bgr(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("opencv-python-headless is required for CLI usage") from e
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Directory of doc_front images")
    ap.add_argument("--onnx", required=True, help="Path to CLIP ONNX model")
    ap.add_argument("--out", required=True, help="Output .npz (ids+vectors)")
    ap.add_argument("--glob", default="*.jpg", help="Image glob within images-dir")
    ap.add_argument(
        "--face-bboxes",
        default=None,
        help="Optional CSV 'id,x1,y1,x2,y2' to mask face region before embedding",
    )
    ap.add_argument("--providers", default="CPUExecutionProvider", help="onnxruntime providers csv")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"images-dir does not exist: {images_dir}")

    face_bbox_map: dict[str, BBoxXYXY] = {}
    if args.face_bboxes:
        import csv

        with open(args.face_bboxes, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _id = str(row["id"])
                face_bbox_map[_id] = BBoxXYXY(
                    x1=int(float(row["x1"])),
                    y1=int(float(row["y1"])),
                    x2=int(float(row["x2"])),
                    y2=int(float(row["y2"])),
                )

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for CLI usage") from e

    providers = [p.strip() for p in str(args.providers).split(",") if p.strip()]
    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    embedder = ClipBackgroundEmbedder(sess)

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for img_path in sorted(images_dir.glob(args.glob)):
        _id = img_path.stem
        img = _load_bgr(img_path)
        if _id in face_bbox_map:
            img = zero_out_bbox(img, face_bbox_map[_id])
        v = embedder.embed_bgr(img)
        ids.append(_id)
        vecs.append(v)

    mat = np.stack(vecs, axis=0) if vecs else np.zeros((0, 512), dtype=np.float32)
    save_vectors(args.out, ids=ids, vectors=mat)
    print(f"Wrote {len(ids)} embeddings to {args.out}")


if __name__ == "__main__":
    main()

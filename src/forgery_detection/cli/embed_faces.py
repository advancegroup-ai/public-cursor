from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forgery_detection.embeddings.face_embedder import FaceEmbedder
from forgery_detection.io.vectors import save_vectors


class _NotImplementedEmbedder:
    def embed_aligned_rgb(self, face_rgb_u8_112: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ArcFace embedding backend is not bundled in this repo. "
            "Plug in a FaceEmbedder implementation (e.g., InsightFace/ORT) and use it here."
        )


def _load_rgb(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("opencv-python-headless is required for CLI usage") from e
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = img[..., ::-1]
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces-dir", required=True, help="Directory of aligned 112x112 face crops")
    ap.add_argument("--out", required=True, help="Output .npz (ids+vectors)")
    ap.add_argument("--glob", default="*.jpg", help="Image glob within faces-dir")
    args = ap.parse_args()

    faces_dir = Path(args.faces_dir)
    if not faces_dir.exists():
        raise SystemExit(f"faces-dir does not exist: {faces_dir}")

    embedder: FaceEmbedder = _NotImplementedEmbedder()

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for img_path in sorted(faces_dir.glob(args.glob)):
        _id = img_path.stem
        rgb = _load_rgb(img_path)
        if rgb.shape[:2] != (112, 112) or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected 112x112x3 uint8, got {rgb.shape} {rgb.dtype} for {img_path}")
        v = np.asarray(embedder.embed_aligned_rgb(rgb), dtype=np.float32)
        if v.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape={v.shape} for {_id}")
        ids.append(_id)
        vecs.append(v)

    dim = int(vecs[0].shape[0]) if vecs else 512
    mat = np.stack(vecs, axis=0) if vecs else np.zeros((0, dim), dtype=np.float32)
    save_vectors(args.out, ids=ids, vectors=mat)
    print(f"Wrote {len(ids)} embeddings to {args.out}")


if __name__ == "__main__":
    main()


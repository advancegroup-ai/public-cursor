from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from atlas_forgery.embeddings.arcface_insightface import ArcFaceConfig, ArcFaceEmbedder
from atlas_forgery.embeddings.doc_front_bg_clip_onnx import ClipOnnxConfig, DocFrontBackgroundClipOnnxEmbedder


def _parse_xyxy(s: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected 'x1,y1,x2,y2'")
    return tuple(int(float(p)) for p in parts)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-front", type=Path, help="Path to doc_front image (BGR readable by OpenCV)")
    ap.add_argument("--liveness", type=Path, help="Path to liveness image (BGR readable by OpenCV)")
    ap.add_argument("--clip-onnx", type=Path, help="Path to CLIP ONNX model for background embedding")
    ap.add_argument("--face-xyxy", type=str, default=None, help="Optional bbox x1,y1,x2,y2 to mask on doc_front")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id for InsightFace ctx_id (use -1 for CPU)")
    args = ap.parse_args()

    if args.doc_front and args.clip_onnx:
        img = cv2.imread(str(args.doc_front))
        if img is None:
            raise SystemExit(f"Failed to read doc_front: {args.doc_front}")
        face_xyxy: Optional[Tuple[int, int, int, int]] = _parse_xyxy(args.face_xyxy) if args.face_xyxy else None
        clip = DocFrontBackgroundClipOnnxEmbedder(ClipOnnxConfig(model_path=args.clip_onnx))
        emb = clip.embed_bgr(img, face_xyxy=face_xyxy, l2_normalize=True)
        print(f"doc_front_bg_clip: shape={emb.shape} dtype={emb.dtype} norm={np.linalg.norm(emb):.4f}")

    if args.liveness:
        img = cv2.imread(str(args.liveness))
        if img is None:
            raise SystemExit(f"Failed to read liveness: {args.liveness}")
        arc = ArcFaceEmbedder(ArcFaceConfig(ctx_id=args.gpu))
        emb = arc.embed_bgr(img, l2_normalize=True)
        if emb is None:
            print("liveness_face_arcface: no face detected")
        else:
            print(f"liveness_face_arcface: shape={emb.shape} dtype={emb.dtype} norm={np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()


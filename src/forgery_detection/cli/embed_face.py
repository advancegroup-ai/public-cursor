from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from forgery_detection.embeddings.face import ArcFaceEmbedder


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed an aligned 112x112 face to a 512-d ArcFace vector.")
    ap.add_argument("--image", required=True, help="Path to aligned face image (112x112).")
    ap.add_argument("--out", required=True, help="Output .npy path.")
    ap.add_argument("--no-l2", action="store_true", help="Disable L2 normalization.")
    args = ap.parse_args()

    # Keep this repo lightweight: only enable real embedding if insightface is installed.
    try:
        import insightface  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"insightface is required for this CLI: {e}")

    # Minimal backend wrapper around insightface model zoo.
    from insightface.model_zoo import model_zoo  # type: ignore

    class _InsightFaceBackend:
        def __init__(self):
            # This resolves a default arcface model; users can swap it in their environment.
            self.model = model_zoo.get_model("arcface_r100_v1")
            self.model.prepare(ctx_id=-1)

        def get_feature(self, aligned_face_rgb: np.ndarray) -> np.ndarray:
            # insightface expects BGR uint8
            bgr = aligned_face_rgb[..., ::-1].astype(np.uint8)
            return self.model.get_feat(bgr).reshape(-1)

    img = Image.open(args.image).convert("RGB")
    arr = np.asarray(img)
    if arr.shape[0] != 112 or arr.shape[1] != 112:
        raise SystemExit(f"Expected 112x112 aligned face, got {arr.shape[1]}x{arr.shape[0]}")

    embedder = ArcFaceEmbedder(backend=_InsightFaceBackend(), l2_normalize=not args.no_l2)
    vec = embedder.embed_aligned(arr)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, vec.astype(np.float32))


if __name__ == "__main__":  # pragma: no cover
    main()


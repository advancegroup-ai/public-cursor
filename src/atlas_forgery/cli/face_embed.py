from __future__ import annotations

import argparse
import csv

import numpy as np

from ..face_embedding import DeterministicFaceEmbedder, InsightFaceArcFaceEmbedder
from ..image_io import load_rgb
from ..vector_store import save_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed aligned 112x112 face crops.")
    ap.add_argument("--csv", required=True, help="CSV with columns: id,path")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--model", choices=["deterministic", "insightface"], default="deterministic")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--path-col", default="path")
    ap.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="insightface onnxruntime provider (when --model=insightface).",
    )
    args = ap.parse_args()

    if args.model == "insightface":
        embedder = InsightFaceArcFaceEmbedder(provider=args.provider)
        model_name = "insightface_arcface_r100_v1"
    else:
        embedder = DeterministicFaceEmbedder()
        model_name = "deterministic"

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    failures: list[tuple[str, str]] = []

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row[args.id_col])
            p = row[args.path_col]
            try:
                img = load_rgb(p).array
                if img.shape[:2] != (112, 112):
                    raise ValueError(f"expected 112x112 aligned crop, got {img.shape}")
                emb = embedder.embed_aligned_112(img)
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


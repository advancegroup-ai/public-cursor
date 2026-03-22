from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from document_face_embeddings.clustering import cluster_sizes, threshold_clusters
from document_face_embeddings.vector_store import load_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold clustering over cosine similarity.")
    ap.add_argument("--vectors-npz", required=True, help="Input .npz with ids + vectors.")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold.")
    ap.add_argument("--output-csv", default=None, help="Optional output CSV: id,label.")
    args = ap.parse_args()

    store = load_npz(args.vectors_npz)
    res = threshold_clusters(store.vectors, threshold=args.threshold)
    sizes = cluster_sizes(res.labels.tolist())
    size_list = sorted(sizes.values(), reverse=True)
    top10 = size_list[:10]
    print(
        f"n={len(store.ids)} dim={store.vectors.shape[1]} threshold={args.threshold} "
        f"clusters={res.n_clusters} top10_sizes={top10}"
    )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("id,label\n")
            for id_, lab in zip(store.ids, res.labels.tolist(), strict=True):
                f.write(f"{id_},{int(lab)}\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()


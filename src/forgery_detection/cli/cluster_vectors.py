from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold
from forgery_detection.io.vectors import load_vectors


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster vectors by cosine threshold (connected components).")
    ap.add_argument("--vectors", required=True, help="Path to vectors file (.npz/.npy/.json/.csv).")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold for edges.")
    ap.add_argument("--out", required=True, help="Output JSON path with clusters.")
    ap.add_argument("--id-col", default="id", help="CSV/JSON list id column.")
    ap.add_argument("--vector-col", default="vector", help="CSV/JSON list vector column.")
    args = ap.parse_args()

    named = load_vectors(args.vectors, id_col=args.id_col, vector_col=args.vector_col)
    res = cluster_by_cosine_threshold([named.vectors[i] for i in range(named.vectors.shape[0])], args.threshold)
    clusters = []
    for members in res.clusters():
        clusters.append([named.ids[i] for i in members])

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(
        json.dumps(
            {"num_vectors": len(named.ids), "num_clusters": res.num_clusters, "threshold": args.threshold, "clusters": clusters},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    main()


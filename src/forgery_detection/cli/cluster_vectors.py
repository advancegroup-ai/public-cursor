from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold
from forgery_detection.io.vectors import load_vectors


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster vectors by cosine threshold.")
    ap.add_argument("--input", required=True, help="Path to .npy/.npz/.json vector file")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold")
    ap.add_argument("--output", required=True, help="Write labels JSON to this path")
    args = ap.parse_args()

    vt = load_vectors(args.input)
    res = cluster_by_cosine_threshold(vt.vectors, threshold=float(args.threshold))

    out = {
        "n": int(vt.vectors.shape[0]),
        "dim": int(vt.vectors.shape[1]),
        "threshold": float(args.threshold),
        "n_clusters": int(res.n_clusters),
        "labels": {vt.ids[i]: int(res.labels[i]) for i in range(len(vt.ids))},
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


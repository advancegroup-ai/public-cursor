from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cluster import threshold_graph_clusters
from ..io import load_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Input vector store .npz")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    store = load_npz(args.npz)
    res = threshold_graph_clusters(store.vectors, threshold=args.threshold)

    out = {
        "n": int(store.vectors.shape[0]),
        "dim": int(store.vectors.shape[1]) if store.vectors.ndim == 2 else None,
        "threshold": float(args.threshold),
        "n_clusters": int(len(res.clusters)),
        "clusters": [
            {"size": int(len(c)), "ids": store.ids[c].tolist(), "indices": c} for c in res.clusters
        ],
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


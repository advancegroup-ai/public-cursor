from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlas_forgery.cluster import threshold_graph_clusters
from atlas_forgery.vector_store import load_npz


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Vector store .npz")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine threshold")
    ap.add_argument("--out-json", required=True, help="Output JSON report path")
    args = ap.parse_args(argv)

    store = load_npz(args.npz)
    res = threshold_graph_clusters(store.vectors, threshold=args.threshold)

    clusters = [
        {"cluster_id": i, "size": len(idxs), "ids": [store.ids[j] for j in idxs]}
        for i, idxs in enumerate(res.clusters)
    ]
    clusters.sort(key=lambda d: d["size"], reverse=True)

    report = {
        "n": len(store.ids),
        "dim": int(store.vectors.shape[1]),
        "threshold": float(args.threshold),
        "num_clusters": len(clusters),
        "clusters": clusters,
        "meta": store.meta,
    }
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


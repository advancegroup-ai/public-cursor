from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..clustering import threshold_graph_clusters
from ..vectors import load_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="VectorStore .npz with ids and vectors")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine sim threshold")
    ap.add_argument("--out", required=True, help="Output JSON report")
    args = ap.parse_args()

    store = load_npz(args.npz)
    res = threshold_graph_clusters(store.vectors, threshold=args.threshold)

    clusters: dict[int, list[str]] = {}
    for _id, lab in zip(store.ids.tolist(), res.labels.tolist(), strict=True):
        clusters.setdefault(int(lab), []).append(str(_id))

    report = {
        "n": int(store.vectors.shape[0]),
        "dim": int(store.vectors.shape[1]),
        "threshold": float(args.threshold),
        "n_clusters": int(res.n_clusters),
        "cluster_sizes": sorted((len(v) for v in clusters.values()), reverse=True),
        "clusters": {str(k): v for k, v in clusters.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


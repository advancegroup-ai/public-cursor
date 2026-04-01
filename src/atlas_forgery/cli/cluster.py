from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..clustering import threshold_graph_clusters
from ..vector_store import load_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster embeddings (cosine-threshold graph).")
    ap.add_argument("--npz", required=True, help="Input embeddings .npz (ids, vectors).")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine threshold for edges.")
    ap.add_argument("--out", required=True, help="Output JSON report path.")
    ap.add_argument("--topk", type=int, default=50, help="Max clusters to include in report.")
    args = ap.parse_args()

    store = load_npz(args.npz)
    ids = store.ids.astype(str)
    vecs = np.asarray(store.vectors, dtype=np.float32)

    res = threshold_graph_clusters(vecs, threshold=args.threshold)
    labels = res.labels
    sizes = res.sizes

    order = np.argsort(-sizes)
    clusters = []
    for rank, k in enumerate(order[: args.topk].tolist(), start=1):
        members = ids[labels == k].tolist()
        clusters.append(
            {
                "rank": rank,
                "cluster_id": int(k),
                "size": int(len(members)),
                "members": members,
            }
        )

    report = {
        "npz": str(Path(args.npz)),
        "n": int(len(ids)),
        "dim": int(vecs.shape[1]) if vecs.ndim == 2 and vecs.shape[0] > 0 else None,
        "threshold": float(args.threshold),
        "num_clusters": int(len(sizes)),
        "largest_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "clusters": clusters,
        "meta": store.meta or {},
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


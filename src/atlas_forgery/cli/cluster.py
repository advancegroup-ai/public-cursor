from __future__ import annotations

import argparse

from ..cluster import threshold_graph_clusters
from ..npz_store import NpzVectorStore


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cluster vectors using cosine threshold connected-components."
    )
    p.add_argument("--npz", required=True, help="Input vectors .npz.")
    p.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold.")
    p.add_argument("--min-size", type=int, default=2, help="Only print clusters >= this size.")
    args = p.parse_args()

    store = NpzVectorStore.load(args.npz)
    clusters = threshold_graph_clusters(store.vectors, threshold=args.threshold)

    kept = [c for c in clusters if len(c) >= args.min_size]
    print(f"Loaded N={len(store.ids)} dim={store.dim} clusters={len(clusters)} kept={len(kept)}")
    for idxs in kept[:200]:
        ids = [store.ids[i] for i in idxs]
        print(f"{len(idxs):>4}  " + " | ".join(ids[:20]))


if __name__ == "__main__":  # pragma: no cover
    main()

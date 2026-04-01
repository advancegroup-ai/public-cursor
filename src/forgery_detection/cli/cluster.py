from __future__ import annotations

import argparse

import numpy as np

from forgery_detection.clustering import connected_components_cosine
from forgery_detection.npz_store import NpzVectorStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster embeddings by cosine threshold.")
    p.add_argument("--npz", required=True, help="Input .npz with ids + vectors.")
    p.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold.")
    p.add_argument("--out-labels", default=None, help="Optional output .npy labels path.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = NpzVectorStore.load(args.npz)
    res = connected_components_cosine(store.vectors, threshold=args.threshold)
    print(f"n={len(store.ids)} clusters={res.n_clusters}")
    if args.out_labels:
        np.save(args.out_labels, res.labels)


if __name__ == "__main__":
    main()

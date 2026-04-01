from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..clustering import cluster_size_stats, cosine_similarity_matrix, threshold_connected_components
from ..vector_store import load_npz


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="Input .npz with ids+vectors")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    args = ap.parse_args(argv)

    store = load_npz(args.vectors)
    sim = cosine_similarity_matrix(store.vectors)
    res = threshold_connected_components(sim, threshold=args.threshold)
    stats = cluster_size_stats(res.labels)

    report = {
        "threshold": float(args.threshold),
        "n_items": int(store.ids.shape[0]),
        "n_clusters": int(res.n_clusters),
        "stats": stats,
        "labels": {str(i): int(l) for i, l in zip(store.ids.tolist(), res.labels.tolist())},
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()


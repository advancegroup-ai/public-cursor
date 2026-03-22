from __future__ import annotations

import argparse
import json

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold
from forgery_detection.io.vectors import load_vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="Path to .npy/.npz/.json vector file")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold")
    ap.add_argument("--out", required=False, help="Optional output json with ids+labels")
    args = ap.parse_args()

    vf = load_vectors(args.vectors)
    res = cluster_by_cosine_threshold(vf.vectors, threshold=args.threshold)

    payload = {
        "n": len(vf.ids),
        "dim": int(vf.vectors.shape[1]),
        "threshold": float(args.threshold),
        "n_clusters": int(res.n_clusters),
        "clusters": {},
    }
    clusters: dict[int, list[str]] = {}
    for _id, lab in zip(vf.ids, res.labels.tolist(), strict=True):
        clusters.setdefault(int(lab), []).append(str(_id))
    payload["clusters"] = {str(k): v for k, v in sorted(clusters.items(), key=lambda kv: kv[0])}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

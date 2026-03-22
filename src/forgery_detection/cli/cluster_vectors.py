from __future__ import annotations

import argparse
import json

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold
from forgery_detection.io.vectors import load_vectors


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--vectors", required=True, help="Path to vectors file (.npy/.npz/.json/.csv)")
  ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold")
  ap.add_argument("--max-clusters", type=int, default=20)
  args = ap.parse_args()

  vs = load_vectors(args.vectors)
  clusters = cluster_by_cosine_threshold(vs.vectors, threshold=args.threshold)

  report = []
  for c in clusters[: args.max_clusters]:
    report.append({"size": len(c), "ids": [vs.ids[i] for i in c]})

  print(json.dumps({"n": len(vs.ids), "threshold": args.threshold, "clusters": report}, ensure_ascii=False))
 

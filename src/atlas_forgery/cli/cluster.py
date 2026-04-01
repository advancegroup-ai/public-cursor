from __future__ import annotations

import argparse
import json

from atlas_forgery.clustering import cluster_sizes, cosine_similarity_matrix, threshold_connected_components
from atlas_forgery.pair_scoring import pairwise_linkage_metrics
from atlas_forgery.vector_store import intersect_ids, load_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster doc embeddings and compute doc-vs-face linkage.")
    ap.add_argument("--doc-npz", required=True)
    ap.add_argument("--face-npz", required=True)
    ap.add_argument("--cluster-threshold", type=float, default=0.92)
    ap.add_argument("--doc-sim-threshold", type=float, default=0.92)
    ap.add_argument("--face-sim-upper", type=float, default=0.45)
    args = ap.parse_args()

    doc = load_npz(args.doc_npz)
    face = load_npz(args.face_npz)
    doc_i, face_i = intersect_ids(doc, face)

    out = {"n_doc": int(doc.ids.shape[0]), "n_face": int(face.ids.shape[0]), "n_intersection": int(doc_i.ids.shape[0])}
    if doc_i.ids.shape[0] == 0:
        out["error"] = "zero_intersection"
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    sim = cosine_similarity_matrix(doc_i.vectors)
    cl = threshold_connected_components(sim, threshold=args.cluster_threshold)
    metric = pairwise_linkage_metrics(
        doc_vectors=doc_i.vectors,
        face_vectors=face_i.vectors,
        doc_sim_threshold=args.doc_sim_threshold,
        face_sim_upper=args.face_sim_upper,
    )
    out["n_clusters"] = int(cl.n_clusters)
    out["cluster_sizes"] = {str(k): int(v) for k, v in cluster_sizes(cl.labels).items()}
    out["pair_metrics"] = {
        "total_pairs": int(metric.total_pairs),
        "linked_pairs": int(metric.linked_pairs),
        "linkage_rate": float(metric.linkage_rate),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

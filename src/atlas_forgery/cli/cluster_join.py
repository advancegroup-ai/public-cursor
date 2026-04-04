from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN

from atlas_forgery.id_normalization import normalize_signature_id
from atlas_forgery.io_npz import load_embeddings_npz


@dataclass(frozen=True)
class JoinStats:
    bg_n: int
    face_n: int
    bg_unique_norm_ids: int
    face_unique_norm_ids: int
    intersection_unique_norm_ids: int


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def join_by_normalized_id(
    bg_ids: list[str],
    bg_vecs: np.ndarray,
    face_ids: list[str],
    face_vecs: np.ndarray,
) -> tuple[list[str], np.ndarray, JoinStats]:
    bg_map: dict[str, int] = {}
    for i, rid in enumerate(bg_ids):
        bg_map[normalize_signature_id(rid).normalized] = i

    face_map: dict[str, int] = {}
    for i, rid in enumerate(face_ids):
        face_map[normalize_signature_id(rid).normalized] = i

    inter = sorted(set(bg_map.keys()) & set(face_map.keys()))
    joined = []
    for k in inter:
        joined.append(k)

    bg_join = np.stack([bg_vecs[bg_map[k]] for k in joined], axis=0)
    face_join = np.stack([face_vecs[face_map[k]] for k in joined], axis=0)
    X = np.concatenate([bg_join, face_join], axis=1)

    stats = JoinStats(
        bg_n=int(bg_vecs.shape[0]),
        face_n=int(face_vecs.shape[0]),
        bg_unique_norm_ids=len(bg_map),
        face_unique_norm_ids=len(face_map),
        intersection_unique_norm_ids=len(inter),
    )
    return joined, X, stats


def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict[str, object]:
    Xn = _l2_normalize(X.astype(np.float32))
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = model.fit_predict(Xn)

    n = int(labels.shape[0])
    n_noise = int((labels == -1).sum())
    clusters = [int(x) for x in sorted(set(labels.tolist())) if x != -1]
    cluster_sizes = {str(c): int((labels == c).sum()) for c in clusters}
    largest = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "n": n,
        "n_noise": n_noise,
        "n_clusters": len(clusters),
        "largest_clusters_top10": largest,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Join bg+face embeddings by normalized signature id and cluster.")
    ap.add_argument("--bg-npz", required=True, help="NPZ with bg embeddings (ids/vectors or bg_ids/bg_vectors)")
    ap.add_argument("--face-npz", required=True, help="NPZ with face embeddings (ids/vectors or face_ids/face_vectors)")
    ap.add_argument("--bg-key", default="default", help="Table key inside bg NPZ (default=default)")
    ap.add_argument("--face-key", default="default", help="Table key inside face NPZ (default=default)")
    ap.add_argument("--dbscan-eps", type=float, default=0.18)
    ap.add_argument("--dbscan-min-samples", type=int, default=3)
    ap.add_argument("--no-cluster", action="store_true", help="Only report intersection stats")
    ap.add_argument("--out-json", default="", help="Write JSON report to file")
    args = ap.parse_args()

    bg_tables = load_embeddings_npz(args.bg_npz)
    face_tables = load_embeddings_npz(args.face_npz)
    if args.bg_key not in bg_tables:
        raise SystemExit(f"bg key {args.bg_key!r} not found. keys={sorted(bg_tables)}")
    if args.face_key not in face_tables:
        raise SystemExit(f"face key {args.face_key!r} not found. keys={sorted(face_tables)}")

    bg = bg_tables[args.bg_key]
    face = face_tables[args.face_key]
    joined_ids, X, stats = join_by_normalized_id(bg.ids, bg.vectors, face.ids, face.vectors)

    report: dict[str, object] = {
        "stats": stats.__dict__,
        "joined_ids_n": len(joined_ids),
        "dim_joined": int(X.shape[1]) if X.size else 0,
    }
    if not args.no_cluster and len(joined_ids) > 0:
        report["dbscan"] = run_dbscan(X, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

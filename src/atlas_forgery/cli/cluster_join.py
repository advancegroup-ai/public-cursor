from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


@dataclass(frozen=True)
class JoinStats:
    n_bg: int
    n_face: int
    n_intersection: int


def _load_npz_embeddings(npz_path: Path, key: str) -> Tuple[List[str], np.ndarray]:
    """
    Accepts either:
    - {key}_ids: array of ids, {key}: (N,D) embeddings
    - or: ids, embeddings (legacy)
    """
    data = np.load(str(npz_path), allow_pickle=True)
    if f"{key}_ids" in data and key in data:
        ids = [str(x) for x in data[f"{key}_ids"].tolist()]
        emb = np.asarray(data[key], dtype=np.float32)
        return ids, emb
    if "ids" in data and "embeddings" in data:
        ids = [str(x) for x in data["ids"].tolist()]
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        return ids, emb
    raise ValueError(f"Unsupported npz format: {npz_path} (expected {key}_ids+{key} or ids+embeddings)")


def join_embeddings(
    bg_ids: List[str],
    bg_emb: np.ndarray,
    face_ids: List[str],
    face_emb: np.ndarray,
) -> Tuple[List[str], np.ndarray, JoinStats]:
    bg_map: Dict[str, int] = {sid: i for i, sid in enumerate(bg_ids)}
    face_map: Dict[str, int] = {sid: i for i, sid in enumerate(face_ids)}
    inter = sorted(set(bg_map).intersection(face_map))
    if not inter:
        return [], np.zeros((0, bg_emb.shape[1] + face_emb.shape[1]), dtype=np.float32), JoinStats(
            n_bg=len(bg_ids), n_face=len(face_ids), n_intersection=0
        )
    joined = np.concatenate([bg_emb[[bg_map[s] for s in inter]], face_emb[[face_map[s] for s in inter]]], axis=1)
    return inter, joined, JoinStats(n_bg=len(bg_ids), n_face=len(face_ids), n_intersection=len(inter))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-npz", type=Path, required=True, help="NPZ containing background embeddings")
    ap.add_argument("--face-npz", type=Path, required=True, help="NPZ containing face embeddings")
    ap.add_argument("--bg-key", type=str, default="bg", help="Embedding key inside bg npz")
    ap.add_argument("--face-key", type=str, default="face", help="Embedding key inside face npz")
    ap.add_argument("--eps", type=float, default=0.15, help="DBSCAN eps on cosine distance")
    ap.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples")
    ap.add_argument("--out-json", type=Path, required=True, help="Output JSON report")
    args = ap.parse_args()

    bg_ids, bg_emb = _load_npz_embeddings(args.bg_npz, args.bg_key)
    face_ids, face_emb = _load_npz_embeddings(args.face_npz, args.face_key)

    joined_ids, joined, stats = join_embeddings(bg_ids, bg_emb, face_ids, face_emb)
    report = {
        "stats": {
            "bg_ids": stats.n_bg,
            "face_ids": stats.n_face,
            "intersection": stats.n_intersection,
            "joined_dim": int(joined.shape[1]),
        },
        "clustering": None,
    }

    if stats.n_intersection >= args.min_samples:
        # DBSCAN expects distances when metric="precomputed"
        dist = cosine_distances(joined)
        cl = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed")
        labels = cl.fit_predict(dist)
        clusters: Dict[int, List[str]] = {}
        for sid, lab in zip(joined_ids, labels):
            clusters.setdefault(int(lab), []).append(sid)
        # Summarize non-noise clusters
        non_noise = {k: v for k, v in clusters.items() if k != -1}
        report["clustering"] = {
            "eps": args.eps,
            "min_samples": args.min_samples,
            "n_clusters_excluding_noise": len(non_noise),
            "noise_count": len(clusters.get(-1, [])),
            "clusters": {str(k): v for k, v in sorted(non_noise.items(), key=lambda kv: -len(kv[1]))[:50]},
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["stats"], indent=2))


if __name__ == "__main__":
    main()


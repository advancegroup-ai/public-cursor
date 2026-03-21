from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


@dataclass(frozen=True)
class JoinStats:
    n_bg: int
    n_face: int
    n_join: int
    join_rate_bg: float
    join_rate_face: float


def _load_npz_map(npz_path: str, *, key_ids: str = "ids", key_emb: str = "embeddings") -> Dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    ids = z[key_ids]
    emb = z[key_emb]
    ids = [str(x) for x in ids.tolist()]
    if len(ids) != emb.shape[0]:
        raise ValueError(f"{npz_path}: ids and embeddings length mismatch")
    return dict(zip(ids, emb.astype(np.float32)))


def join_embeddings(bg: Dict[str, np.ndarray], face: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, JoinStats]:
    bg_ids = set(bg.keys())
    face_ids = set(face.keys())
    common = sorted(bg_ids & face_ids)
    rows = []
    for sid in common:
        rows.append(
            {
                "signature_id": sid,
                "bg_emb": bg[sid],
                "face_emb": face[sid],
            }
        )
    df = pd.DataFrame(rows)
    stats = JoinStats(
        n_bg=len(bg_ids),
        n_face=len(face_ids),
        n_join=len(common),
        join_rate_bg=(len(common) / max(1, len(bg_ids))),
        join_rate_face=(len(common) / max(1, len(face_ids))),
    )
    return df, stats


def cluster_dbscan(vectors: np.ndarray, *, eps: float, min_samples: int, metric: str = "cosine") -> np.ndarray:
    if metric == "cosine":
        # DBSCAN expects a distance metric; for cosine we pass precomputed distances for stability.
        d = cosine_distances(vectors)
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        return model.fit_predict(d)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    return model.fit_predict(vectors)


def main() -> None:
    ap = argparse.ArgumentParser(description="Join bg+face embeddings by signature_id and run DBSCAN.")
    ap.add_argument("--bg-npz", required=True, help="NPZ with ids + embeddings for background (doc_front)")
    ap.add_argument("--face-npz", required=True, help="NPZ with ids + embeddings for face (liveness or doc face)")
    ap.add_argument("--bg-eps", type=float, default=0.15, help="DBSCAN eps for bg (cosine distance)")
    ap.add_argument("--face-eps", type=float, default=0.35, help="DBSCAN eps for face (cosine distance)")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--out-json", default=None, help="Optional path to write summary JSON")
    args = ap.parse_args()

    bg = _load_npz_map(args.bg_npz)
    face = _load_npz_map(args.face_npz)
    df, stats = join_embeddings(bg, face)
    if stats.n_join == 0:
        summary = {
            "join": stats.__dict__,
            "error": "no_overlap",
            "hint": "Ensure both NPZ files use same signature_id set and same id key name.",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    bg_mat = np.stack(df["bg_emb"].to_numpy(), axis=0)
    face_mat = np.stack(df["face_emb"].to_numpy(), axis=0)

    bg_labels = cluster_dbscan(bg_mat, eps=args.bg_eps, min_samples=args.min_samples, metric="cosine")
    face_labels = cluster_dbscan(face_mat, eps=args.face_eps, min_samples=args.min_samples, metric="cosine")
    df["bg_cluster"] = bg_labels
    df["face_cluster"] = face_labels

    # Report cluster sizes (exclude noise label -1)
    def _sizes(labels: np.ndarray) -> List[Tuple[int, int]]:
        s = pd.Series(labels)
        s = s[s != -1]
        vc = s.value_counts().sort_values(ascending=False)
        return [(int(k), int(v)) for k, v in vc.head(20).items()]

    summary = {
        "join": stats.__dict__,
        "bg": {
            "eps": args.bg_eps,
            "min_samples": args.min_samples,
            "n_clusters": int(len(set(bg_labels)) - (1 if -1 in bg_labels else 0)),
            "noise": int(np.sum(bg_labels == -1)),
            "top_sizes": _sizes(bg_labels),
        },
        "face": {
            "eps": args.face_eps,
            "min_samples": args.min_samples,
            "n_clusters": int(len(set(face_labels)) - (1 if -1 in face_labels else 0)),
            "noise": int(np.sum(face_labels == -1)),
            "top_sizes": _sizes(face_labels),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from atlas_forgery.id_normalization import normalize_signature_id
from atlas_forgery.io_npz import load_embedding_npz


def _join_tables(bg_npz: Path, face_npz: Path) -> tuple[pd.DataFrame, dict]:
    bg = load_embedding_npz(bg_npz)
    face = load_embedding_npz(face_npz)

    bg_norm = [normalize_signature_id(x).normalized for x in bg.ids]
    face_norm = [normalize_signature_id(x).normalized for x in face.ids]

    bg_df = pd.DataFrame(
        {"bg_raw_id": bg.ids, "signature_id": bg_norm, "bg_row": np.arange(len(bg.ids))}
    )
    face_df = pd.DataFrame(
        {
            "face_raw_id": face.ids,
            "signature_id": face_norm,
            "face_row": np.arange(len(face.ids)),
        }
    )

    # Keep only rows with parseable normalized signature_id
    bg_df = bg_df[bg_df["signature_id"].notna()].copy()
    face_df = face_df[face_df["signature_id"].notna()].copy()

    joined = bg_df.merge(face_df, on="signature_id", how="inner")

    stats = {
        "bg_total": len(bg.ids),
        "face_total": len(face.ids),
        "bg_parseable": int(bg_df.shape[0]),
        "face_parseable": int(face_df.shape[0]),
        "intersection_pairs": int(joined.shape[0]),
        "intersection_unique_signature_ids": int(joined["signature_id"].nunique())
        if not joined.empty
        else 0,
        "bg_dim": int(bg.vectors.shape[1]),
        "face_dim": int(face.vectors.shape[1]),
    }
    return joined, stats


def _cluster(joined: pd.DataFrame, bg_npz: Path, face_npz: Path, eps: float, min_samples: int, metric: str) -> tuple[pd.DataFrame, dict]:
    bg = load_embedding_npz(bg_npz)
    face = load_embedding_npz(face_npz)

    if joined.empty:
        return joined.assign(cluster=-1), {"clusters": 0, "noise": 0}

    bg_vec = bg.vectors[joined["bg_row"].to_numpy()]
    face_vec = face.vectors[joined["face_row"].to_numpy()]

    # L2-normalize then concatenate (cosine in DBSCAN works as 1 - cosine_sim)
    bg_vec = normalize(bg_vec, norm="l2")
    face_vec = normalize(face_vec, norm="l2")
    X = np.concatenate([bg_vec, face_vec], axis=1)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = model.fit_predict(X)

    out = joined.copy()
    out["cluster"] = labels
    clusters = sorted([c for c in set(labels.tolist()) if c != -1])
    summary = {
        "clusters": len(clusters),
        "noise": int((labels == -1).sum()),
        "cluster_sizes": out[out["cluster"] != -1]["cluster"].value_counts().to_dict(),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Join bg+face embeddings by normalized signature_id, then (optionally) DBSCAN cluster.")
    ap.add_argument("--bg-npz", required=True, help="NPZ with background embeddings (ids + vectors).")
    ap.add_argument("--face-npz", required=True, help="NPZ with face embeddings (ids + vectors).")
    ap.add_argument("--out-csv", required=True, help="Output CSV with join (and cluster labels).")
    ap.add_argument("--out-json", required=True, help="Output JSON summary (stats + clustering).")
    ap.add_argument("--cluster", action="store_true", help="Run DBSCAN clustering on concatenated normalized vectors.")
    ap.add_argument("--eps", type=float, default=0.15, help="DBSCAN eps (distance threshold).")
    ap.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples.")
    ap.add_argument("--metric", type=str, default="cosine", help="DBSCAN metric (default cosine).")
    args = ap.parse_args()

    bg_npz = Path(args.bg_npz)
    face_npz = Path(args.face_npz)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    joined, stats = _join_tables(bg_npz, face_npz)

    cluster_summary: dict = {}
    if args.cluster:
        joined2, cluster_summary = _cluster(
            joined=joined,
            bg_npz=bg_npz,
            face_npz=face_npz,
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
        )
        joined = joined2

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_csv, index=False)

    summary = {"stats": stats, "clustering": cluster_summary}
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class JoinClusterConfig:
    id_col: str = "signature_id"
    bg_vec_col: str = "bg_vec"
    face_vec_col: str = "face_vec"
    eps_bg: float = 0.12
    min_samples_bg: int = 2
    eps_face: float = 0.45
    min_samples_face: int = 2


def _load_npz_vecs(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k])
    return out


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    return x


def run_join_cluster(
    bg_npz: Path,
    face_npz: Path,
    out_csv: Path,
    out_json: Path,
    cfg: JoinClusterConfig,
) -> dict:
    bg = _load_npz_vecs(bg_npz)
    face = _load_npz_vecs(face_npz)

    bg_ids = set(bg.keys())
    face_ids = set(face.keys())
    joined = sorted(bg_ids & face_ids)

    summary: dict = {
        "bg_count": len(bg_ids),
        "face_count": len(face_ids),
        "intersection": len(joined),
    }

    if not joined:
        out_csv.write_text("")
        out_json.write_text(json.dumps(summary, indent=2))
        return summary

    bg_mat = np.stack([_ensure_2d(bg[i]).reshape(-1) for i in joined], axis=0)
    face_mat = np.stack([_ensure_2d(face[i]).reshape(-1) for i in joined], axis=0)

    bg_labels = DBSCAN(eps=cfg.eps_bg, min_samples=cfg.min_samples_bg, metric="cosine").fit_predict(bg_mat)
    face_labels = DBSCAN(eps=cfg.eps_face, min_samples=cfg.min_samples_face, metric="cosine").fit_predict(face_mat)

    df = pd.DataFrame(
        {
            cfg.id_col: joined,
            "bg_cluster": bg_labels.astype(int),
            "face_cluster": face_labels.astype(int),
        }
    )
    df.to_csv(out_csv, index=False)

    def _cluster_stats(labels: np.ndarray) -> dict:
        labels = np.asarray(labels)
        n_noise = int(np.sum(labels == -1))
        clusters = labels[labels >= 0]
        uniq, counts = np.unique(clusters, return_counts=True) if clusters.size else (np.array([]), np.array([]))
        return {
            "n_noise": n_noise,
            "n_clusters": int(len(uniq)),
            "largest_cluster": int(counts.max()) if counts.size else 0,
        }

    summary.update(
        {
            "bg": _cluster_stats(bg_labels),
            "face": _cluster_stats(face_labels),
            "out_csv": str(out_csv),
        }
    )
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Join background+face embeddings by ID and run DBSCAN clustering.")
    p.add_argument("--bg-npz", type=Path, required=True, help="NPZ with {id -> 512d background embedding}")
    p.add_argument("--face-npz", type=Path, required=True, help="NPZ with {id -> 512d face embedding}")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--eps-bg", type=float, default=JoinClusterConfig.eps_bg)
    p.add_argument("--min-samples-bg", type=int, default=JoinClusterConfig.min_samples_bg)
    p.add_argument("--eps-face", type=float, default=JoinClusterConfig.eps_face)
    p.add_argument("--min-samples-face", type=int, default=JoinClusterConfig.min_samples_face)
    args = p.parse_args(argv)

    cfg = JoinClusterConfig(
        eps_bg=args.eps_bg,
        min_samples_bg=args.min_samples_bg,
        eps_face=args.eps_face,
        min_samples_face=args.min_samples_face,
    )
    run_join_cluster(
        bg_npz=args.bg_npz,
        face_npz=args.face_npz,
        out_csv=args.out_csv,
        out_json=args.out_json,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()


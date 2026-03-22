from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_forgery.cli.cluster_join import join_by_normalized_id, run_dbscan


def main() -> None:
    # Synthetic embeddings with mixed ID formats to validate normalization + join.
    bg_ids = [
        "2b7f2f3f-1c2d-4d0b-8fe2-8f0f1a2b3c4d",
        "sig:2b7f2f3f-1c2d-4d0b-8fe2-8f0f1a2b3c4e",
        "signature_id=2b7f2f3f-1c2d-4d0b-8fe2-8f0f1a2b3c4f",
        "UNJOINABLE_001",
    ]
    face_ids = [
        "uid:2b7f2f3f-1c2d-4d0b-8fe2-8f0f1a2b3c4d",
        "2b7f2f3f-1c2d-4d0b-8fe2-8f0f1a2b3c4e",
        "xxx 2b7f2f3f-1c2d-4d0b-8fe2-8f0f1a2b3c4f yyy",
        "UNJOINABLE_999",
    ]

    rng = np.random.default_rng(0)
    bg_vecs = rng.normal(size=(len(bg_ids), 512)).astype(np.float32)
    face_vecs = rng.normal(size=(len(face_ids), 512)).astype(np.float32)

    joined_ids, X, stats = join_by_normalized_id(bg_ids, bg_vecs, face_ids, face_vecs)
    report = {
        "intersection_unique_signature_ids": stats.intersection_unique_norm_ids,
        "joined_ids": joined_ids,
        "dim_joined": int(X.shape[1]),
        "dbscan": run_dbscan(X, eps=0.3, min_samples=2),
    }

    Path("tmp").mkdir(exist_ok=True)
    Path("tmp/smoke_cluster_join_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from forgery_detection.clustering.threshold_unionfind import cluster_by_cosine_threshold


def _load_vectors(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix == ".npy":
        return np.load(p)

    if p.suffix == ".npz":
        z = np.load(p)
        if "vectors" in z:
            return z["vectors"]
        # fallback to first key
        keys = list(z.keys())
        if not keys:
            raise ValueError("Empty npz file")
        return z[keys[0]]

    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "vectors" in data:
            data = data["vectors"]
        return np.asarray(data, dtype=np.float32)

    if p.suffix == ".csv":
        rows: list[list[float]] = []
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                rows.append([float(x) for x in row])
        return np.asarray(rows, dtype=np.float32)

    raise ValueError(f"Unsupported input format: {p.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster vectors by cosine threshold (single-linkage).")
    ap.add_argument("--vectors", required=True, help="Input vectors (.npy/.npz/.json/.csv)")
    ap.add_argument("--threshold", type=float, required=True, help="Cosine similarity threshold")
    ap.add_argument("--out", default="-", help="Output labels json path or '-' for stdout")
    args = ap.parse_args()

    vecs = _load_vectors(args.vectors)
    labels = cluster_by_cosine_threshold(vecs, threshold=args.threshold)

    payload = {"n": int(len(labels)), "threshold": float(args.threshold), "labels": labels}
    text = json.dumps(payload, ensure_ascii=False)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()


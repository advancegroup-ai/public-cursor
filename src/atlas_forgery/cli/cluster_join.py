import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


def _load_npz(path: Path, key: str) -> dict[str, np.ndarray]:
     data = np.load(path, allow_pickle=False)
     if key not in data:
         raise KeyError(f"missing key '{key}' in {path}; keys={list(data.keys())}")
     arr = data[key]
     if arr.dtype != np.float32 and arr.dtype != np.float64:
         arr = arr.astype(np.float32, copy=False)
     ids_key = f"{key}_ids"
     if ids_key not in data:
         raise KeyError(f"missing key '{ids_key}' in {path}; expected paired id array")
     ids = data[ids_key].astype(str)
     return {str(i): arr[idx] for idx, i in enumerate(ids)}
 
 
def main() -> None:
     p = argparse.ArgumentParser(description="Join bg+face embeddings and cluster on concatenation.")
     p.add_argument("--npz", type=str, required=True, help="NPZ containing embedding arrays + *_ids arrays.")
     p.add_argument("--bg-key", type=str, default="doc_front_bg", help="Key for background embedding matrix.")
     p.add_argument("--face-key", type=str, default="liveness_face", help="Key for face embedding matrix.")
     p.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps.")
     p.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples.")
     p.add_argument("--out-dir", type=str, required=True, help="Output directory.")
     args = p.parse_args()
 
     out_dir = Path(args.out_dir)
     out_dir.mkdir(parents=True, exist_ok=True)
 
     npz_path = Path(args.npz)
     bg = _load_npz(npz_path, args.bg_key)
     face = _load_npz(npz_path, args.face_key)
 
     common = sorted(set(bg.keys()) & set(face.keys()))
     summary = {
         "npz": str(npz_path),
         "bg_key": args.bg_key,
         "face_key": args.face_key,
         "bg_n": len(bg),
         "face_n": len(face),
         "intersection_n": len(common),
     }
 
     if not common:
         (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
         return
 
     X = np.stack([np.concatenate([bg[i], face[i]]).astype(np.float32, copy=False) for i in common], axis=0)
     X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
 
     labels = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="euclidean").fit_predict(X)
     summary["dbscan"] = {
         "eps": args.eps,
         "min_samples": args.min_samples,
         "n_points": int(X.shape[0]),
         "n_clusters_excl_noise": int(len(set(labels)) - (1 if -1 in labels else 0)),
         "noise_frac": float(np.mean(labels == -1)),
     }
 
     rows = [{"id": cid, "label": int(lbl)} for cid, lbl in zip(common, labels)]
     (out_dir / "labels.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
     (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
 
 
if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _write_npz(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=object), vectors=vectors)


def main() -> None:
    tmp = Path("tmp/smoke")
    bg_npz = tmp / "bg.npz"
    face_npz = tmp / "face.npz"
    out_csv = tmp / "joined.csv"
    out_json = tmp / "summary.json"

    rng = np.random.default_rng(0)
    bg_ids = ["sig_123456", "signature_id=234567", "no_digits", "345678.jpg"]
    face_ids = ["123456.png", "face-234567", "345678", "999999"]

    _write_npz(bg_npz, bg_ids, rng.normal(size=(len(bg_ids), 4)).astype("float32"))
    _write_npz(face_npz, face_ids, rng.normal(size=(len(face_ids), 5)).astype("float32"))

    import subprocess
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "atlas_forgery.cli.cluster_join",
            "--bg-npz",
            str(bg_npz),
            "--face-npz",
            str(face_npz),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ]
    )

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    print("summary:", json.dumps(summary, indent=2))
    assert summary["stats"]["intersection_unique_signature_ids"] == 3


if __name__ == "__main__":
    main()


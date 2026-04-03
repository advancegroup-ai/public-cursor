import json
from pathlib import Path

import numpy as np


def main() -> None:
    out = Path("tmp_smoke")
    out.mkdir(parents=True, exist_ok=True)
 
    ids_bg = np.array(["a", "b", "c", "d"])
    ids_face = np.array(["b", "c", "d", "e"])
    rng = np.random.default_rng(0)
 
    bg = rng.normal(size=(len(ids_bg), 512)).astype(np.float32)
    face = rng.normal(size=(len(ids_face), 512)).astype(np.float32)
 
    np.savez(
        out / "embeddings.npz",
        doc_front_bg=bg,
        doc_front_bg_ids=ids_bg,
        liveness_face=face,
        liveness_face_ids=ids_face,
    )
 
    meta = {
        "npz": str(out / "embeddings.npz"),
        "expected_intersection": sorted(set(ids_bg.tolist()) & set(ids_face.tolist())),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
 
 
if __name__ == "__main__":
    main()

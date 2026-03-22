import argparse
import json
from pathlib import Path

import numpy as np

from atlas_forgery.embeddings.arcface_insightface import ArcFaceInsightFaceEmbedder
from atlas_forgery.embeddings.doc_front_bg_clip_onnx import DocFrontBgClipOnnxEmbedder


def _np_to_list(x: np.ndarray) -> list[float]:
     return [float(v) for v in x.reshape(-1)]
 
 
def main() -> None:
     p = argparse.ArgumentParser(description="Embed a single doc_front and/or face image.")
     p.add_argument("--doc-front", type=str, default=None, help="Path to doc_front image.")
     p.add_argument("--liveness", type=str, default=None, help="Path to liveness/selfie image.")
     p.add_argument("--bg-onnx", type=str, default=None, help="Path to CLIP ONNX for background embedding.")
     p.add_argument(
         "--mask-face-on-doc",
         action="store_true",
         help="Attempt face detection on doc_front and zero-out face region before embedding.",
     )
     p.add_argument("--out", type=str, required=True, help="Output JSON path.")
     args = p.parse_args()
 
     out_path = Path(args.out)
     out_path.parent.mkdir(parents=True, exist_ok=True)
 
     out: dict = {"doc_front": None, "liveness": None}
 
     if args.doc_front and args.bg_onnx:
         bg = DocFrontBgClipOnnxEmbedder(onnx_path=args.bg_onnx)
         vec = bg.embed(Path(args.doc_front), mask_face=args.mask_face_on_doc)
         out["doc_front"] = {"embedding_512": _np_to_list(vec)}
 
     if args.liveness:
         face = ArcFaceInsightFaceEmbedder()
         vec = face.embed(Path(args.liveness))
         out["liveness"] = {"embedding_512": _np_to_list(vec)}
 
     out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
 
 
if __name__ == "__main__":
    main()

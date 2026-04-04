 #!/usr/bin/env python3
 from __future__ import annotations
 
 import argparse
 import json
 from pathlib import Path
 
 import cv2
 import numpy as np
 
 from atlas_forgery.embeddings.arcface_insightface import ArcFaceEmbedder
 from atlas_forgery.embeddings.doc_front_bg_clip_onnx import ClipBgEmbedderONNX, detect_face_bbox_insightface
 
 
 def main() -> int:
     ap = argparse.ArgumentParser(description="Embed doc_front background (face-masked) and liveness face.")
     ap.add_argument("--doc-front", required=True, help="Path to doc_front image")
     ap.add_argument("--liveness", required=False, help="Path to liveness image (optional)")
     ap.add_argument("--clip-onnx", required=True, help="Path to CLIP bg embedding ONNX")
     ap.add_argument("--no-face-mask", action="store_true", help="Disable face masking on doc_front")
     args = ap.parse_args()
 
     doc_path = Path(args.doc_front)
     doc = cv2.imread(str(doc_path))
     if doc is None:
         raise SystemExit(f"Failed to read doc_front: {doc_path}")
 
     face_bbox = None if args.no_face_mask else detect_face_bbox_insightface(doc)
     clip = ClipBgEmbedderONNX(args.clip_onnx)
     doc_emb = clip.embed(doc, face_bbox=face_bbox)
 
     out = {
         "doc_front": {
             "path": str(doc_path),
             "face_bbox": None
             if face_bbox is None
             else {"x1": face_bbox.x1, "y1": face_bbox.y1, "x2": face_bbox.x2, "y2": face_bbox.y2},
             "embedding_dim": int(doc_emb.shape[0]),
             "embedding_l2": float(np.linalg.norm(doc_emb)),
             "embedding_head": [float(x) for x in doc_emb[:8]],
         }
     }
 
     if args.liveness:
         live_path = Path(args.liveness)
         live = cv2.imread(str(live_path))
         if live is None:
             raise SystemExit(f"Failed to read liveness: {live_path}")
         arc = ArcFaceEmbedder()
         res = arc.embed_largest_face(live)
         if res is None:
             out["liveness"] = {"path": str(live_path), "face_found": False}
         else:
             out["liveness"] = {
                 "path": str(live_path),
                 "face_found": True,
                 "det_score": float(res.det_score),
                 "bbox_xyxy": [float(x) for x in res.bbox_xyxy.tolist()],
                 "embedding_dim": int(res.embedding.shape[0]),
                 "embedding_l2": float(np.linalg.norm(res.embedding)),
                 "embedding_head": [float(x) for x in res.embedding[:8]],
             }
 
     print(json.dumps(out, ensure_ascii=False, indent=2))
     return 0
 
 
 if __name__ == "__main__":
     raise SystemExit(main())

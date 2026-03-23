import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .embeddings import (
    ArcFaceEmbedder,
    ClipOnnxEmbedder,
    EmbedConfig,
    embed_doc_front_background,
)
from .io import ImageRecord, iter_images_from_dir, load_image_bgr
from .masking import FaceMasker


def _write_npz(out_path: Path, arr: np.ndarray, ids: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, embeddings=arr.astype(np.float32), ids=np.array(ids))


def cmd_embed_doc_front(args: argparse.Namespace) -> int:
    img_dir = Path(args.image_dir)
    out = Path(args.out_npz)
    model_path = args.clip_onnx

    cfg = EmbedConfig(clip_onnx_path=model_path, providers=args.providers)
    clip = ClipOnnxEmbedder(cfg)
    masker = FaceMasker(
        face_cascade_path=args.haar_face_xml,
        min_face_size=args.min_face_size,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
    )

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    failures: list[dict] = []

    for rec in iter_images_from_dir(img_dir, exts=args.exts):
        try:
            bgr = load_image_bgr(rec.path)
            emb = embed_doc_front_background(bgr, clip=clip, masker=masker, return_meta=False)
            ids.append(rec.signature_id)
            vecs.append(emb)
        except Exception as e:
            failures.append({"path": str(rec.path), "error": repr(e)})

    if not vecs:
        raise SystemExit(f"No embeddings produced from {img_dir}")

    mat = np.stack(vecs, axis=0)
    _write_npz(out, mat, ids)

    if args.out_failures_json:
        Path(args.out_failures_json).write_text(json.dumps(failures, indent=2))

    print(
        json.dumps(
            {
                "mode": "doc_front_bg",
                "n_images": len(ids),
                "n_failures": len(failures),
                "out_npz": str(out),
            },
            indent=2,
        )
    )
    return 0


def cmd_embed_face(args: argparse.Namespace) -> int:
    img_dir = Path(args.image_dir)
    out = Path(args.out_npz)

    embedder = ArcFaceEmbedder(
        model_name=args.model_name,
        providers=args.providers,
    )

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    failures: list[dict] = []

    for rec in iter_images_from_dir(img_dir, exts=args.exts):
        try:
            bgr = load_image_bgr(rec.path)
            emb = embedder.embed(bgr)
            ids.append(rec.signature_id)
            vecs.append(emb)
        except Exception as e:
            failures.append({"path": str(rec.path), "error": repr(e)})

    if not vecs:
        raise SystemExit(f"No face embeddings produced from {img_dir}")

    mat = np.stack(vecs, axis=0)
    _write_npz(out, mat, ids)

    if args.out_failures_json:
        Path(args.out_failures_json).write_text(json.dumps(failures, indent=2))

    print(
        json.dumps(
            {
                "mode": "face",
                "n_images": len(ids),
                "n_failures": len(failures),
                "out_npz": str(out),
            },
            indent=2,
        )
    )
    return 0


def _load_npz(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    d = np.load(npz_path, allow_pickle=False)
    embs = d["embeddings"].astype(np.float32)
    ids = [str(x) for x in d["ids"].tolist()]
    return embs, ids


def cmd_eval_join(args: argparse.Namespace) -> int:
    bg_embs, bg_ids = _load_npz(Path(args.bg_npz))
    face_embs, face_ids = _load_npz(Path(args.face_npz))

    bg = pd.DataFrame({"signature_id": bg_ids})
    bg["bg_idx"] = np.arange(len(bg_ids))
    face = pd.DataFrame({"signature_id": face_ids})
    face["face_idx"] = np.arange(len(face_ids))

    joined = bg.merge(face, on="signature_id", how="inner")
    stats = {
        "bg_n": len(bg_ids),
        "face_n": len(face_ids),
        "intersection_n": int(len(joined)),
        "intersection_ratio_bg": float(len(joined) / max(1, len(bg_ids))),
        "intersection_ratio_face": float(len(joined) / max(1, len(face_ids))),
    }

    if args.out_join_csv:
        out_csv = Path(args.out_join_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(out_csv, index=False)
        stats["out_join_csv"] = str(out_csv)

    print(json.dumps(stats, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas-doc-forgery")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bg = sub.add_parser("embed-doc-front-bg", help="CLIP background embedding with face masking")
    p_bg.add_argument("--image-dir", required=True)
    p_bg.add_argument("--out-npz", required=True)
    p_bg.add_argument("--clip-onnx", required=True, help="Path to CLIP ONNX model")
    p_bg.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"])
    p_bg.add_argument("--haar-face-xml", default=None, help="Optional custom haarcascade path")
    p_bg.add_argument("--min-face-size", type=int, default=40)
    p_bg.add_argument("--scale-factor", type=float, default=1.1)
    p_bg.add_argument("--min-neighbors", type=int, default=5)
    p_bg.add_argument("--exts", nargs="*", default=[".jpg", ".jpeg", ".png", ".webp"])
    p_bg.add_argument("--out-failures-json", default=None)
    p_bg.set_defaults(func=cmd_embed_doc_front)

    p_face = sub.add_parser("embed-face", help="ArcFace/InsightFace 512-d face embedding")
    p_face.add_argument("--image-dir", required=True)
    p_face.add_argument("--out-npz", required=True)
    p_face.add_argument("--model-name", default="buffalo_l")
    p_face.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"])
    p_face.add_argument("--exts", nargs="*", default=[".jpg", ".jpeg", ".png", ".webp"])
    p_face.add_argument("--out-failures-json", default=None)
    p_face.set_defaults(func=cmd_embed_face)

    p_join = sub.add_parser("eval-join", help="Compute signature_id overlap between two NPZ files")
    p_join.add_argument("--bg-npz", required=True)
    p_join.add_argument("--face-npz", required=True)
    p_join.add_argument("--out-join-csv", default=None)
    p_join.set_defaults(func=cmd_eval_join)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


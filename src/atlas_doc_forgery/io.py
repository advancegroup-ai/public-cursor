from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImageRecord:
    signature_id: str
    path: Path


def normalize_signature_id(value: object) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if raw.endswith(".0") and raw.replace(".", "", 1).isdigit():
        raw = raw[:-2]
    return raw.lower()


def infer_signature_id_from_path(path: Path) -> str:
    return normalize_signature_id(path.stem)


def iter_images_from_dir(image_dir: Path, exts: Iterable[str]) -> list[ImageRecord]:
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    out: list[ImageRecord] = []
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in ext_set:
            out.append(ImageRecord(signature_id=infer_signature_id_from_path(p), path=p))
    return out


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


@dataclass
class Sample:
    signature_id: str
    doc_front_path: str
    liveness_path: str | None = None


def load_cases_csv(csv_path: str, image_root: str | None = None) -> list[Sample]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    sig_col = cols.get("signature_id") or cols.get("signatureid") or cols.get("uid")
    doc_col = cols.get("doc_front") or cols.get("doc_front_path") or cols.get("document_front")
    liv_col = cols.get("liveness") or cols.get("liveness_path") or cols.get("face_image")
    if not sig_col or not doc_col:
        raise ValueError("CSV must include signature_id and doc_front path columns")

    root = Path(image_root) if image_root else None
    samples: list[Sample] = []
    for _, row in df.iterrows():
        sig = normalize_signature_id(row[sig_col])
        doc = str(row[doc_col]).strip()
        liv = str(row[liv_col]).strip() if liv_col and pd.notna(row[liv_col]) else None
        if root:
            doc = str((root / doc).resolve()) if not Path(doc).is_absolute() else doc
            if liv:
                liv = str((root / liv).resolve()) if not Path(liv).is_absolute() else liv
        samples.append(Sample(signature_id=sig, doc_front_path=doc, liveness_path=liv))
    return samples

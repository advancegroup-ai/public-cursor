from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceMaskResult:
    masked_bgr: np.ndarray
    face_bbox: Optional[tuple[int, int, int, int]]  # x1,y1,x2,y2


class FaceMasker:
    """Detects a face and zeros out its pixels on the input image."""

    def __init__(
        self,
        face_cascade_path: str | None = None,
        min_face_size: int = 40,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
    ) -> None:
        cascade_path = face_cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.min_face_size = int(min_face_size)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)

    def _detect_face_bbox(self, image_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)[0]
        return (int(x), int(y), int(x + w), int(y + h))

    def mask(self, image_bgr: np.ndarray) -> FaceMaskResult:
        out = image_bgr.copy()
        bbox = self._detect_face_bbox(out)
        if bbox is None:
            return FaceMaskResult(masked_bgr=out, face_bbox=None)
        x1, y1, x2, y2 = bbox
        out[y1:y2, x1:x2] = 0
        return FaceMaskResult(masked_bgr=out, face_bbox=bbox)


@dataclass(frozen=True)
class MaskingResult:
    masked_image_bgr: np.ndarray
    doc_bbox: Optional[tuple[int, int, int, int]]  # x1,y1,x2,y2
    face_bbox_on_doc: Optional[tuple[int, int, int, int]]  # x1,y1,x2,y2 in global coords


def _clamp_bbox(
    bbox: tuple[int, int, int, int], *, w: int, h: int
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def detect_document_bbox(image_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Heuristic card boundary detection:
    - edge detection + contour ranking
    - choose largest plausible rectangle
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = 0.1 * h * w
    best = None
    best_area = 0.0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        ratio = bw / max(bh, 1)
        if ratio < 1.2 or ratio > 2.6:
            continue
        if area > best_area:
            best_area = area
            best = (int(x), int(y), int(x + bw), int(y + bh))
    return best


def mask_face_on_document(image_bgr: np.ndarray) -> MaskingResult:
    """
    Requested masking rule:
      img[y1:y2, x1:x2] = 0
    Face is detected on a doc crop if doc bbox is found, otherwise on full image.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    doc_bbox = detect_document_bbox(out)

    if doc_bbox is not None:
        dx1, dy1, dx2, dy2 = doc_bbox
        doc_crop = out[dy1:dy2, dx1:dx2]
        face_doc = FaceMasker()._detect_face_bbox(doc_crop)
        if face_doc is not None:
            fx1, fy1, fx2, fy2 = face_doc
            global_face = _clamp_bbox((dx1 + fx1, dy1 + fy1, dx1 + fx2, dy1 + fy2), w=w, h=h)
            if global_face is not None:
                x1, y1, x2, y2 = global_face
                out[y1:y2, x1:x2] = 0
                return MaskingResult(out, doc_bbox, global_face)

    face_full = FaceMasker()._detect_face_bbox(out)
    if face_full is not None:
        face_full = _clamp_bbox(face_full, w=w, h=h)
        if face_full is not None:
            x1, y1, x2, y2 = face_full
            out[y1:y2, x1:x2] = 0
            return MaskingResult(out, doc_bbox, face_full)

    return MaskingResult(out, doc_bbox, None)


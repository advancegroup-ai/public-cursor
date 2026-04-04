from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


_DIGITS_RE = re.compile(r"(\d{6,})")


@dataclass(frozen=True)
class NormalizedId:
    raw: str
    normalized: Optional[str]


def normalize_signature_id(value: object) -> NormalizedId:
    """
    Normalize heterogeneous signature-id representations to a stable join key.

    Common failure mode: bg embeddings keyed by "sig_123" while face embeddings
    keyed by "123.jpg" or "signature_id=123" => naive string join yields 0 overlap.
    Strategy: extract a long digit run (>=6 digits) as canonical ID, else None.
    """
    if value is None:
        return NormalizedId(raw="None", normalized=None)
    raw = str(value).strip()
    if raw == "":
        return NormalizedId(raw=raw, normalized=None)
    m = _DIGITS_RE.search(raw)
    if not m:
        return NormalizedId(raw=raw, normalized=None)
    return NormalizedId(raw=raw, normalized=m.group(1))


def normalize_many(values: Iterable[object]) -> list[NormalizedId]:
    return [normalize_signature_id(v) for v in values]


def unique_normalized(values: Iterable[object]) -> set[str]:
    out: set[str] = set()
    for v in values:
        n = normalize_signature_id(v).normalized
        if n is not None:
            out.add(n)
    return out


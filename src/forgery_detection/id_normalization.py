from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class IdNormalizer:
    """
    Normalize signature/document ids coming from different pipelines.

    Goal: maximize deterministic overlaps between two id lists by applying safe transforms.
    """

    strip: bool = True
    lowercase: bool = True
    collapse_whitespace: bool = True
    drop_common_prefixes: tuple[str, ...] = (
        "signature_id=",
        "sig=",
        "uid=",
        "id=",
    )

    def normalize_one(self, raw: str) -> str:
        s = raw
        if self.strip:
            s = s.strip()
        if self.collapse_whitespace:
            s = re.sub(r"\s+", " ", s)
        if self.lowercase:
            s = s.lower()
        for p in self.drop_common_prefixes:
            if s.startswith(p):
                s = s[len(p) :]
                if self.strip:
                    s = s.strip()
                break
        return s

    def normalize_many(self, raws: Iterable[str]) -> list[str]:
        return [self.normalize_one(r) for r in raws]


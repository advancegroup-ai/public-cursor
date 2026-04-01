from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class IdNormalization:
    trim: bool = True
    collapse_ws: bool = True
    lower: bool = False
    strip_prefixes: tuple[str, ...] = ()

    def normalize(self, raw: object) -> str:
        s = str(raw)
        if self.trim:
            s = s.strip()
        if self.collapse_ws:
            s = _WS.sub("", s)
        for p in self.strip_prefixes:
            if s.startswith(p):
                s = s[len(p) :]
        if self.lower:
            s = s.lower()
        return s


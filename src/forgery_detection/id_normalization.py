from __future__ import annotations

import re
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class IdNormalizer:
    """
    Normalizes IDs from different embedding pipelines so they can be joined.

    This intentionally stays conservative (no hashing) and is easy to reason about.
    """
    lowercase: bool = True
    collapse_whitespace: bool = True
    strip: bool = True
    strip_prefixes: tuple[str, ...] = ("sig:", "signature:", "uid:", "id:")

    def normalize(self, raw: str) -> str:
        s = raw
        if self.strip:
            s = s.strip()
        if self.collapse_whitespace:
            s = _WS_RE.sub(" ", s)
        if self.lowercase:
            s = s.lower()

        for p in self.strip_prefixes:
            if s.startswith(p):
                s = s[len(p) :]
                if self.strip:
                    s = s.strip()
                break
        return s

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_PREFIXES = ("signature_id=", "signature=", "sig=", "uid=")
 
 
@dataclass(frozen=True)
class IdNormalizer:
    """
    Normalizes ids so two pipelines with slightly different id formatting can be joined.
    This intentionally stays conservative and deterministic.
    """
 
    lower: bool = True
    strip_prefixes: bool = True
    collapse_ws: bool = True
 
    def normalize(self, raw: str) -> str:
        s = str(raw).strip()
        if self.collapse_ws:
            s = _WS.sub(" ", s)
        if self.strip_prefixes:
            s_lower = s.lower()
            for p in _PREFIXES:
                if s_lower.startswith(p):
                    s = s[len(p) :].strip()
                    break
        if self.lower:
            s = s.lower()
        return s

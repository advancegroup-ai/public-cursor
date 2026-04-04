import re
from dataclasses import dataclass


_HEX_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


@dataclass(frozen=True)
class NormalizedId:
    raw: str
    normalized: str


def normalize_signature_id(value: object) -> NormalizedId:
    """
    Normalize heterogeneous signature_id strings into a stable join key.

    Common variants observed in logs/artifacts:
    - raw UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    - prefixed like "signature_id=UUID", "sig:UUID", "uid:UUID"
    - embedded in longer strings
    """
    s = "" if value is None else str(value)
    s_stripped = s.strip()

    if _HEX_RE.match(s_stripped):
        return NormalizedId(raw=s, normalized=s_stripped.lower())

    # Extract first UUID-looking token.
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", s_stripped, re.I)
    if m:
        return NormalizedId(raw=s, normalized=m.group(1).lower())

    # Fallback: alnum compacting for stable-ish join.
    fallback = re.sub(r"[^0-9a-zA-Z]+", "", s_stripped).lower()
    return NormalizedId(raw=s, normalized=fallback)

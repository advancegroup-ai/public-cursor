from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgery_detection.id_normalization import IdNormalizer
from forgery_detection.npz_store import NpzVectorStore, join_on_normalized_ids


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forgery-npz-join",
        description="Join two NPZ vector stores on normalized ids and report overlap metrics.",
    )
    p.add_argument("--left", required=True, help="Left .npz file containing arrays: ids, vectors")
    p.add_argument("--right", required=True, help="Right .npz file containing arrays: ids, vectors")
    p.add_argument("--out-left", default="", help="Optional output path for joined left store")
    p.add_argument("--out-right", default="", help="Optional output path for joined right store")
    p.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing in id normalization",
    )
    p.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Add a common prefix to drop during normalization (can repeat)",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    normalizer = IdNormalizer(
        lowercase=not args.no_lowercase,
        drop_common_prefixes=tuple(IdNormalizer().drop_common_prefixes) + tuple(args.drop_prefix),
    )

    left = NpzVectorStore.load(args.left)
    right = NpzVectorStore.load(args.right)
    left_j, right_j, report = join_on_normalized_ids(left, right, normalizer=normalizer)

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.out_left:
        left_j.save(Path(args.out_left))
    if args.out_right:
        right_j.save(Path(args.out_right))


if __name__ == "__main__":
    main()


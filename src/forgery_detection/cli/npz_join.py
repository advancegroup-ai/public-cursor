from __future__ import annotations

import argparse
from dataclasses import asdict

from forgery_detection.id_normalization import IdNormalizer
from forgery_detection.join import join_on_normalized_ids
from forgery_detection.npz_store import NpzVectorStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Join two embedding NPZ stores by normalized IDs.")
    p.add_argument("--left", required=True, help="Left .npz path (must contain ids + vectors).")
    p.add_argument("--right", required=True, help="Right .npz path (must contain ids + vectors).")
    p.add_argument("--out-left", default=None, help="Write joined-left store to this .npz.")
    p.add_argument("--out-right", default=None, help="Write joined-right store to this .npz.")
    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercase normalization.")
    p.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Additional prefix to strip (can repeat).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    left = NpzVectorStore.load(args.left)
    right = NpzVectorStore.load(args.right)
    normalizer = IdNormalizer(
        lowercase=not args.no_lowercase,
        strip_prefixes=tuple(IdNormalizer().strip_prefixes) + tuple(args.strip_prefix),
    )
    l2, r2, report = join_on_normalized_ids(left, right, normalizer=normalizer)
    print(asdict(report))
    if args.out_left:
        l2.save(args.out_left)
    if args.out_right:
        r2.save(args.out_right)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys

from forgery_detection.id_normalize import IdNormalization
from forgery_detection.join import join_on_normalized_ids
from forgery_detection.npz_store import NpzVectorStore


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forgery-npz-join",
        description=(
            "Inner-join two NPZ vector stores by normalized IDs and optionally "
            "write aligned outputs."
        ),
    )
    p.add_argument("--left", required=True, help="Left NPZ path (contains ids, vectors).")
    p.add_argument("--right", required=True, help="Right NPZ path (contains ids, vectors).")
    p.add_argument("--left-name", default="left")
    p.add_argument("--right-name", default="right")
    p.add_argument("--out-left", default=None, help="Write aligned left store NPZ to this path.")
    p.add_argument("--out-right", default=None, help="Write aligned right store NPZ to this path.")
    p.add_argument("--strip-prefix", action="append", default=[], help="Strip prefix(es) from IDs.")
    p.add_argument("--lower", action="store_true", help="Lowercase IDs during normalization.")
    p.add_argument("--no-trim", action="store_true", help="Do not trim whitespace.")
    p.add_argument("--keep-ws", action="store_true", help="Do not remove internal whitespace.")
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--json", action="store_true", help="Emit report as JSON only.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    left = NpzVectorStore.load(args.left)
    right = NpzVectorStore.load(args.right)

    norm = IdNormalization(
        trim=not args.no_trim,
        collapse_ws=not args.keep_ws,
        lower=bool(args.lower),
        strip_prefixes=tuple(args.strip_prefix),
    )

    left_aligned, right_aligned, report = join_on_normalized_ids(
        left,
        right,
        left_name=args.left_name,
        right_name=args.right_name,
        norm=norm,
        max_examples=args.max_examples,
    )

    if args.out_left:
        left_aligned.save(args.out_left)
    if args.out_right:
        right_aligned.save(args.out_right)

    payload = {
        "left": report.left_name,
        "right": report.right_name,
        "left_n": report.left_n,
        "right_n": report.right_n,
        "overlap_n": report.overlap_n,
        "overlap_ratio_left": report.overlap_ratio_left,
        "overlap_ratio_right": report.overlap_ratio_right,
        "left_only_n": report.left_only_n,
        "right_only_n": report.right_only_n,
        "examples_left_only": report.examples_left_only,
        "examples_right_only": report.examples_right_only,
        "out_left_n": left_aligned.size,
        "out_right_n": right_aligned.size,
        "out_dim_left": left_aligned.dim,
        "out_dim_right": right_aligned.dim,
    }

    if args.json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write("Join report: ")
        sys.stdout.write(
            f"{report.left_name}({report.left_n}) vs {report.right_name}({report.right_n})\n"
        )
        sys.stdout.write(
            "- overlap: "
            f"{report.overlap_n} (left {report.overlap_ratio_left:.2%}, "
            f"right {report.overlap_ratio_right:.2%})\n"
        )
        sys.stdout.write(f"- left_only: {report.left_only_n}, right_only: {report.right_only_n}\n")
        if report.examples_left_only:
            sys.stdout.write(f"- examples_left_only: {report.examples_left_only[:5]}\n")
        if report.examples_right_only:
            sys.stdout.write(f"- examples_right_only: {report.examples_right_only[:5]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


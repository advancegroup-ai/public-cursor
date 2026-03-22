from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ..npz_store import NpzVectorStore


@dataclass(frozen=True)
class JoinedStores:
    ids: list[str]
    a: np.ndarray
    b: np.ndarray


def join_by_id(a: NpzVectorStore, b: NpzVectorStore) -> JoinedStores:
    a_index = {sid: i for i, sid in enumerate(a.ids)}
    b_index = {sid: i for i, sid in enumerate(b.ids)}
    common = sorted(set(a_index) & set(b_index))
    if common:
        a_vec = np.stack([a.vectors[a_index[sid]] for sid in common], axis=0)
        b_vec = np.stack([b.vectors[b_index[sid]] for sid in common], axis=0)
    else:
        a_vec = a.vectors[:0]
        b_vec = b.vectors[:0]
    return JoinedStores(ids=common, a=a_vec, b=b_vec)


def main() -> None:
    p = argparse.ArgumentParser(description="Join two NPZ vector stores by intersecting ids.")
    p.add_argument("--a", required=True, help="First .npz (e.g., doc_front background embeddings).")
    p.add_argument("--b", required=True, help="Second .npz (e.g., liveness face embeddings).")
    p.add_argument("--out-a", default=None, help="Optional output .npz for subset of A.")
    p.add_argument("--out-b", default=None, help="Optional output .npz for subset of B.")
    args = p.parse_args()

    a = NpzVectorStore.load(args.a)
    b = NpzVectorStore.load(args.b)
    joined = join_by_id(a, b)

    print(
        "Join summary:\n"
        f"- A: N={len(a.ids)} dim={a.dim}\n"
        f"- B: N={len(b.ids)} dim={b.dim}\n"
        f"- intersection: N={len(joined.ids)}"
    )

    if args.out_a:
        NpzVectorStore(ids=joined.ids, vectors=joined.a).save(args.out_a)
        print(f"Wrote subset A -> {args.out_a}")
    if args.out_b:
        NpzVectorStore(ids=joined.ids, vectors=joined.b).save(args.out_b)
        print(f"Wrote subset B -> {args.out_b}")


if __name__ == "__main__":  # pragma: no cover
    main()

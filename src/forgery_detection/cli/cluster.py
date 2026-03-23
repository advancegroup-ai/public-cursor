import argparse

from forgery_detection.cluster import cluster_by_cosine_threshold
from forgery_detection.vector_store import NpzVectorStore


def main() -> None:
   parser = argparse.ArgumentParser(
       description="Cluster embeddings with cosine similarity threshold (connected components)."
   )
   parser.add_argument("--npz", required=True, help="Input .npz (ids + vectors).")
   parser.add_argument(
       "--threshold", type=float, required=True, help="Cosine similarity threshold."
   )
   parser.add_argument(
       "--min-size", type=int, default=2, help="Only print clusters >= this size."
   )
   args = parser.parse_args()

   store = NpzVectorStore.load(args.npz)
   clusters = cluster_by_cosine_threshold(store.ids, store.vectors, threshold=args.threshold)

   big = [c for c in clusters if len(c.member_indices) >= args.min_size]
   print(f"clusters_total={len(clusters)} clusters_ge_{args.min_size}={len(big)}")
   for c in big[:50]:
       members = [store.ids[i] for i in c.member_indices]
       print(f"cluster_id={c.cluster_id} size={len(members)} members={members}")


if __name__ == "__main__":
   main()

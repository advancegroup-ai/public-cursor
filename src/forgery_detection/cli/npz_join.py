import argparse
from dataclasses import asdict

from forgery_detection.join import join_on_normalized_ids
from forgery_detection.vector_store import NpzVectorStore


def main() -> None:
   parser = argparse.ArgumentParser(
       description="Join two .npz embedding stores on normalized ids."
   )
   parser.add_argument("--left", required=True, help="Path to left .npz (ids + vectors).")
   parser.add_argument("--right", required=True, help="Path to right .npz (ids + vectors).")
   parser.add_argument(
       "--out-left", default=None, help="Optional output .npz for joined left store."
   )
   parser.add_argument(
       "--out-right", default=None, help="Optional output .npz for joined right store."
   )
   args = parser.parse_args()

   left = NpzVectorStore.load(args.left)
   right = NpzVectorStore.load(args.right)
   joined = join_on_normalized_ids(left, right)

   print("join_report=" + str(asdict(joined.report)))

   if args.out_left:
       joined.left.save(args.out_left)
   if args.out_right:
       joined.right.save(args.out_right)


if __name__ == "__main__":
   main()

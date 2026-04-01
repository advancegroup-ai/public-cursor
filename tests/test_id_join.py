import numpy as np

from forgery_detection.id_normalization import IdNormalizer
from forgery_detection.npz_store import NpzVectorStore, join_on_normalized_ids


def test_join_on_normalized_ids_matches_after_prefix_and_case() -> None:
    left = NpzVectorStore(
        ids=np.array(["signature_id=ABC ", "signature_id=def"], dtype=object),
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    right = NpzVectorStore(
        ids=np.array(["abc", " DEF "], dtype=object),
        vectors=np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32),
    )
    normalizer = IdNormalizer()
    left_j, right_j, report = join_on_normalized_ids(left, right, normalizer=normalizer)

    assert len(left_j) == 2
    assert len(right_j) == 2
    assert report.overlap_unique_norm == 2


def test_join_first_occurrence_wins() -> None:
    left = NpzVectorStore(
        ids=np.array(["a", "A", "b"], dtype=object),
        vectors=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    )
    right = NpzVectorStore(
        ids=np.array(["a", "b", "B"], dtype=object),
        vectors=np.array([[10.0], [30.0], [40.0]], dtype=np.float32),
    )
    left_j, right_j, _ = join_on_normalized_ids(left, right)

    # "a" on left chooses index 0 not 1; "b" chooses index 2; on right chooses first b (index 1).
    assert left_j.vectors[:, 0].tolist() == [1.0, 3.0]
    assert right_j.vectors[:, 0].tolist() == [10.0, 30.0]


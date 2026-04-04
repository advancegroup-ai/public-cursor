import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold
 
 
def test_cluster_by_cosine_threshold_connected_components():
  # Two tight clusters in 2D
  a = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
  b = np.array([[0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
  x = np.vstack([a, b])
  clusters = cluster_by_cosine_threshold(x, threshold=0.98)
  assert sorted([sorted(c) for c in clusters]) == [[0, 1], [2, 3]]

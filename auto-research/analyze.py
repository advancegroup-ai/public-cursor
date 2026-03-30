"""
Liveness detection analysis — the file the agent modifies.
Experiment 10: 1 feature per image (noise only, 2 total) + RF 50.

Testing if Laplacian noise alone suffices.

Usage: python analyze.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from prepare import (
    load_dataset, load_image, get_train_test_split,
    evaluate, print_results, print_detailed_report,
)

# ---------------------------------------------------------------------------
# Feature extraction (MODIFY THIS)
# ---------------------------------------------------------------------------

def _minimal_features(img):
    """1 feature: Laplacian noise level."""
    gray = img.mean(axis=2)

    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (gray[:-2, 1:-1] + gray[2:, 1:-1] +
                        gray[1:-1, :-2] + gray[1:-1, 2:] -
                        4 * gray[1:-1, 1:-1])
    noise = float(np.abs(lap).mean())

    return [noise]


def extract_features(sample: dict) -> np.ndarray:
    """Extract features from a single sample."""
    features = []
    far = load_image(sample["far"])
    near = load_image(sample["near"])

    for img in [far, near]:
        features.extend(_minimal_features(img))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification (MODIFY THIS)
# ---------------------------------------------------------------------------

def build_classifier():
    """Build and return a classifier."""
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Main pipeline (MODIFY THIS)
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    samples, labels = load_dataset()
    print(f"Loaded {len(samples)} samples")

    print("Splitting train/test...")
    train_samples, train_labels, test_samples, test_labels = get_train_test_split(
        samples, labels
    )
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    print("Extracting features...")
    X_train = np.array([extract_features(s) for s in train_samples])
    X_test = np.array([extract_features(s) for s in test_samples])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=-1.0)

    num_features = X_train.shape[1]
    print(f"Features: {num_features}")

    print("Training classifier...")
    clf = build_classifier()
    clf.fit(X_train, train_labels)

    print("Evaluating...")
    y_pred = clf.predict(X_test)

    metrics = evaluate(test_labels, y_pred)
    print_results(metrics, num_features, len(samples))
    print_detailed_report(test_labels, y_pred)


if __name__ == "__main__":
    main()

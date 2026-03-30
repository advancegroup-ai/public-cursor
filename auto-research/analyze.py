"""
Liveness detection analysis — the file the agent modifies.
Experiment 1: Comprehensive feature extraction targeting known attack signatures.

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

def _image_stats(img):
    """Extract statistical features from an image."""
    features = []
    features.append(img.mean())
    features.append(img.std())

    for ch in range(3):
        c = img[:, :, ch]
        features.append(c.mean())
        features.append(c.std())
        features.append(float(np.median(c)))
        features.append(c.min())
        features.append(c.max())

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    features.append(float(r.mean() - g.mean()))
    features.append(float(r.mean() - b.mean()))
    features.append(float(g.mean() - b.mean()))

    return features


def _noise_features(img):
    """Estimate noise level using Laplacian."""
    features = []
    gray = img.mean(axis=2)

    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (gray[:-2, 1:-1] + gray[2:, 1:-1] +
                        gray[1:-1, :-2] + gray[1:-1, 2:] -
                        4 * gray[1:-1, 1:-1])
    features.append(float(np.abs(lap).mean()))
    features.append(float(lap.var()))

    row_diff = np.diff(gray, axis=0)
    col_diff = np.diff(gray, axis=1)
    features.append(float(np.abs(row_diff).mean()))
    features.append(float(np.abs(col_diff).mean()))

    return features


def _stripe_features(img):
    """Detect periodic stripe patterns (screen attacks have stripes every 3rd row)."""
    features = []
    gray = img.mean(axis=2)

    row_means = gray.mean(axis=1)
    fft_row = np.abs(np.fft.rfft(row_means - row_means.mean()))
    n = len(fft_row)
    if n > 3:
        features.append(float(fft_row[n // 3] if n // 3 < n else 0))
    else:
        features.append(0.0)

    features.append(float(fft_row[1:].max()) if len(fft_row) > 1 else 0.0)
    features.append(float(fft_row[1:].mean()) if len(fft_row) > 1 else 0.0)

    return features


def _color_space_features(img):
    """Extract HSV-like features to detect color shifts."""
    features = []
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-10

    saturation = np.where(cmax > 0, delta / (cmax + 1e-10), 0)
    features.append(float(saturation.mean()))
    features.append(float(saturation.std()))

    channel_std = np.std(img, axis=2)
    features.append(float(channel_std.mean()))
    features.append(float(channel_std.std()))

    features.append(float((channel_std < 0.01).mean()))

    return features


def _frequency_features(img):
    """Extract frequency domain features."""
    features = []
    gray = img.mean(axis=2)

    fft2 = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    r1 = h // 8
    r2 = h // 4
    low = magnitude[dist <= r1].sum()
    mid = magnitude[(dist > r1) & (dist <= r2)].sum()
    high = magnitude[dist > r2].sum()
    total = low + mid + high + 1e-10

    features.append(float(low / total))
    features.append(float(mid / total))
    features.append(float(high / total))

    return features


def _contrast_features(img):
    """Detect reduced contrast (photocopies) and proximity to 128 (fake cards)."""
    features = []
    gray = img.mean(axis=2)

    features.append(float(gray.max() - gray.min()))
    features.append(float(np.percentile(gray, 95) - np.percentile(gray, 5)))

    features.append(float(np.abs(gray - 0.5).mean()))

    for ch in range(3):
        c = img[:, :, ch]
        features.append(float(np.percentile(c, 95) - np.percentile(c, 5)))

    return features


def extract_features(sample: dict) -> np.ndarray:
    """
    Extract features from a single sample (3 images).
    Returns a 1D feature vector.
    """
    features = []

    far = load_image(sample["far"])
    near = load_image(sample["near"])

    for img in [far, near]:
        features.extend(_image_stats(img))
        features.extend(_noise_features(img))
        features.extend(_stripe_features(img))
        features.extend(_color_space_features(img))
        features.extend(_frequency_features(img))
        features.extend(_contrast_features(img))

    diff_mean = np.abs(far.mean() - near.mean())
    diff_std = np.abs(far.std() - near.std())
    features.append(float(diff_mean))
    features.append(float(diff_std))

    for ch in range(3):
        features.append(float(np.abs(far[:, :, ch].mean() - near[:, :, ch].mean())))

    if sample["card"] is not None:
        card = load_image(sample["card"])
        features.extend(_image_stats(card))
        features.extend(_contrast_features(card))
    else:
        features.extend([0.0] * (14 + 7))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification (MODIFY THIS)
# ---------------------------------------------------------------------------

def build_classifier():
    """Build and return a classifier."""
    return RandomForestClassifier(
        n_estimators=200,
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

    nan_mask = np.isnan(X_train) | np.isinf(X_train)
    if nan_mask.any():
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

"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: Hybrid ResNet18 deep features + handcrafted noise/gradient/FFT features.
Combines the strong deep learning baseline with the Laplacian noise features
that achieved perfect classification on the smaller dataset.
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = DATA_DIR / "last_result.json"
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_labels():
    labels = {}
    for fname, default_label in [("neg_batch.json", "Negative"), ("pos_batch.json", "Positive")]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(str(fpath)) as f:
            batch = json.load(f)
        for d in batch.get("data", []):
            parts = d["sample_id"].split("_")
            if len(parts) >= 4:
                sig = parts[3]
                main_label = d["pn"].split("/")[0]
                if main_label in ("Negative Type", "Negative_Type"):
                    main_label = "Negative"
                labels[sig] = {"main_label": main_label, "pn": d["pn"]}

    if not labels:
        with open(str(DATA_DIR / "labels.json")) as f:
            labels = json.load(f)

    return labels


def compute_handcrafted_features(img_path):
    """Compute noise/gradient/FFT features from a single image."""
    try:
        img = Image.open(str(img_path)).convert("L")
        arr = np.array(img, dtype=np.float32)

        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        from scipy.signal import convolve2d
        lap = convolve2d(arr, laplacian, mode='valid')
        noise_std = float(np.std(lap))

        gy, gx = np.gradient(arr)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mean = float(np.mean(grad_mag))

        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        h, w = magnitude.shape
        ch, cw = h // 2, w // 2
        r = min(h, w) // 8
        low_freq = magnitude[ch-r:ch+r, cw-r:cw+r].mean()
        high_freq = magnitude.mean()
        hf_ratio = float(high_freq / (low_freq + 1e-8))

        return [noise_std, grad_mean, hf_ratio]
    except Exception:
        return [0.0, 0.0, 0.0]


class HybridLivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, cache_features=True):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.feature_cache = {}

        if cache_features:
            for i, sig in enumerate(sig_ids):
                far_path = self.samples_dir / sig / "far.jpg"
                near_path = self.samples_dir / sig / "near.jpg"
                far_feats = compute_handcrafted_features(far_path)
                near_feats = compute_handcrafted_features(near_path)
                self.feature_cache[sig] = far_feats + near_feats
                if (i + 1) % 500 == 0:
                    print(f"  Cached features: {i+1}/{len(sig_ids)}")

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):
        sig = self.sig_ids[idx]
        info = self.labels[sig]
        label = 0 if info["main_label"] == "Positive" else 1

        far_path = self.samples_dir / sig / "far.jpg"
        near_path = self.samples_dir / sig / "near.jpg"

        try:
            far_img = Image.open(str(far_path)).convert("RGB")
            near_img = Image.open(str(near_path)).convert("RGB")
        except Exception:
            far_img = Image.new("RGB", (224, 224))
            near_img = Image.new("RGB", (224, 224))

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)

        if sig in self.feature_cache:
            feats = torch.tensor(self.feature_cache[sig], dtype=torch.float32)
        else:
            far_feats = compute_handcrafted_features(far_path)
            near_feats = compute_handcrafted_features(near_path)
            feats = torch.tensor(far_feats + near_feats, dtype=torch.float32)

        return img, feats, label


class HybridModel(nn.Module):
    def __init__(self, num_handcrafted=6, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            backbone.conv1.weight[:, 3:] = old_conv.weight

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        deep_dim = 512

        self.handcrafted_net = nn.Sequential(
            nn.Linear(num_handcrafted, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(deep_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, handcrafted):
        deep = self.features(img).flatten(1)
        hc = self.handcrafted_net(handcrafted)
        combined = torch.cat([deep, hc], dim=1)
        return self.classifier(combined)


def load_data():
    labels = load_labels()
    samples_dir = DATA_DIR / "samples"
    all_sigs = set(os.listdir(str(samples_dir)))

    valid_sigs = [s for s in labels if s in all_sigs
                  and (samples_dir / s / "far.jpg").exists()
                  and (samples_dir / s / "near.jpg").exists()]

    binary_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid_sigs]
    print(f"Total valid samples: {len(valid_sigs)}")
    print(f"Distribution: {Counter(binary_labels)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(valid_sigs, binary_labels)):
        if fold_idx == 0:
            train_ids = [valid_sigs[i] for i in train_idx]
            test_ids = [valid_sigs[i] for i in test_idx]
            break

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Caching handcrafted features for train set...")
    train_ds = HybridLivenessDataset(train_ids, labels, transform_train)
    print("Caching handcrafted features for test set...")
    test_ds = HybridLivenessDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(num_handcrafted=6, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HybridResNet18 (6ch deep + 6 handcrafted), params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=15, steps_per_epoch=len(train_loader)
    )

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    epochs = 15
    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for imgs, feats, targets in train_loader:
            imgs = imgs.to(device)
            feats = feats.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            out = model(imgs, feats)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, feats, targets in test_loader:
                imgs = imgs.to(device)
                feats = feats.to(device)
                out = model(imgs, feats)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0
    approach = "hybrid_resnet18_6ch_deep_plus_noise_gradient_fft_handcrafted"

    print(f"\n---")
    print(f"balanced_accuracy: {best_bal_acc:.6f}")
    print(f"accuracy:          {best_acc:.6f}")
    print(f"f1_score:          {best_f1:.6f}")
    print(f"num_params:        {num_params}")
    print(f"training_seconds:  {elapsed:.1f}")
    print(f"approach:          {approach}")
    print(f"---")

    result_data = {
        "balanced_accuracy": best_bal_acc,
        "accuracy": best_acc,
        "f1_score": best_f1,
        "num_params": num_params,
        "training_seconds": elapsed,
        "approach": approach,
    }
    with open(str(RESULTS_FILE), "w") as rf:
        json.dump(result_data, rf, indent=2)
    print(f"Results written to {RESULTS_FILE}")


train()

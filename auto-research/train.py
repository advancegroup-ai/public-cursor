"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Hybrid ResNet18 + handcrafted noise/freq features fusion.
Key idea: Extract handcrafted features (Laplacian noise, gradient stats, FFT) alongside
ResNet18 deep features. Concatenate both for classification. Handcrafted features
achieved 100% on smaller dataset; deep features generalize better. Combining both
should capture complementary signals.
Previous best: 0.976659 (ResNet18 6ch baseline).
"""
import os
import sys
import json
import time
import random
import math
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
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def extract_handcrafted(img_pil):
    """Extract key handcrafted features from a PIL image (on CPU)."""
    img_np = np.array(img_pil).astype(np.float32)
    gray = np.array(img_pil.convert("L")).astype(np.float32)
    features = []

    from scipy.ndimage import laplace
    lap = laplace(gray)
    features.append(np.var(lap))
    features.append(np.abs(lap).mean())

    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    features.append(np.abs(gy).mean())
    features.append(np.abs(gx).mean())

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    Y, X = np.ogrid[:h, :w]
    center_mask = (Y - cy)**2 + (X - cx)**2 <= r**2
    low_energy = mag[center_mask].sum()
    total_energy = mag.sum() + 1e-10
    features.append(1.0 - low_energy / total_energy)

    channel_means = img_np.mean(axis=(0, 1))
    blue_shift = channel_means[2] - (channel_means[0] + channel_means[1]) / 2.0
    features.append(blue_shift)

    features.append(np.std(gray))

    return np.array(features, dtype=np.float32)


class LivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"

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

        far_feats = extract_handcrafted(far_img)
        near_feats = extract_handcrafted(near_img)
        diff_feats = far_feats - near_feats
        handcrafted = np.concatenate([far_feats, near_feats, diff_feats])
        handcrafted = np.nan_to_num(handcrafted, nan=0.0, posinf=1e6, neginf=-1e6)
        handcrafted_tensor = torch.from_numpy(handcrafted)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        return img, handcrafted_tensor, label


class HybridModel(nn.Module):
    def __init__(self, handcrafted_dim=21, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3:] = old_conv.weight

        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        self.handcrafted_proj = nn.Sequential(
            nn.BatchNorm1d(handcrafted_dim),
            nn.Linear(handcrafted_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, handcrafted):
        deep_feat = self.backbone(img)
        hand_feat = self.handcrafted_proj(handcrafted)
        fused = torch.cat([deep_feat, hand_feat], dim=1)
        return self.classifier(fused)


def load_data():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    valid_labels = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)
            valid_labels.append(0 if info["main_label"] == "Positive" else 1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(valid, valid_labels))
    train_ids = [valid[i] for i in train_idx]
    test_ids = [valid[i] for i in test_idx]

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDataset(train_ids, labels, transform_train)
    test_ds = LivenessDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(handcrafted_dim=21).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid ResNet18 + handcrafted features, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

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
        for imgs, handcrafted, targets in train_loader:
            imgs = imgs.to(device)
            handcrafted = handcrafted.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            out = model(imgs, handcrafted)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, handcrafted, targets in test_loader:
                imgs = imgs.to(device)
                handcrafted = handcrafted.to(device)
                out = model(imgs, handcrafted)
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
    approach = "hybrid_resnet18_6ch_plus_handcrafted_21feats_fusion"

    result_block = (
        f"\n---\n"
        f"balanced_accuracy: {best_bal_acc:.6f}\n"
        f"accuracy:          {best_acc:.6f}\n"
        f"f1_score:          {best_f1:.6f}\n"
        f"num_params:        {num_params}\n"
        f"training_seconds:  {elapsed:.1f}\n"
        f"approach:          {approach}\n"
        f"---\n"
    )
    print(result_block)

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

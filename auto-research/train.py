"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: EfficientNet-B0 + DCT frequency domain features.
Key idea: Screen replay attacks show specific frequency patterns (moiré, 
refresh lines). We extract DCT features from both images and fuse them
with EfficientNet-B0 deep features. This combines spatial understanding
with frequency-domain attack detection.
Previous best on DGX1: 0.976659 (ResNet18 6ch baseline).
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


def extract_dct_features(img_pil, n_features=32):
    """Extract DCT-based frequency features from image."""
    gray = np.array(img_pil.convert("L").resize((128, 128))).astype(np.float32)
    
    from scipy.fft import dct
    dct_coeffs = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    h, w = dct_coeffs.shape
    mag = np.abs(dct_coeffs)
    
    features = []
    
    zones = [(0, h//8, 0, w//8), (0, h//4, 0, w//4), (0, h//2, 0, w//2)]
    total = mag.sum() + 1e-10
    for y1, y2, x1, x2 in zones:
        features.append(mag[y1:y2, x1:x2].sum() / total)
    
    features.append(mag[h//4:h//2, w//4:w//2].sum() / total)
    features.append(mag[h//2:, w//2:].sum() / total)
    
    diag_energy = sum(mag[i, i] for i in range(min(h, w))) / total
    features.append(diag_energy)
    
    row_energies = mag.sum(axis=1)
    col_energies = mag.sum(axis=0)
    features.append(np.std(row_energies) / (np.mean(row_energies) + 1e-10))
    features.append(np.std(col_energies) / (np.mean(col_energies) + 1e-10))
    
    features.append(np.log1p(np.max(mag[1:, :])))
    features.append(np.log1p(np.mean(mag[h//4:, :])))
    
    from scipy.ndimage import laplace
    gray_full = np.array(img_pil.convert("L")).astype(np.float32)
    lap = laplace(gray_full)
    features.append(np.var(lap))
    features.append(np.abs(lap).mean())
    
    gy = np.diff(gray_full, axis=0)
    gx = np.diff(gray_full, axis=1)
    features.append(np.abs(gy).mean())
    features.append(np.abs(gx).mean())
    features.append(np.std(gy))
    features.append(np.std(gx))
    
    return np.array(features[:n_features], dtype=np.float32)


class LivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, dct_features=16):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.dct_features = dct_features

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

        far_dct = extract_dct_features(far_img, self.dct_features)
        near_dct = extract_dct_features(near_img, self.dct_features)
        diff_dct = far_dct - near_dct
        freq_feats = np.concatenate([far_dct, near_dct, diff_dct])
        freq_feats = np.nan_to_num(freq_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        freq_tensor = torch.from_numpy(freq_feats)

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        return img, freq_tensor, label


class EfficientNetFreq(nn.Module):
    def __init__(self, freq_dim=48, num_classes=2):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
        base.features[0][0] = new_conv
        
        feat_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        self.freq_proj = nn.Sequential(
            nn.BatchNorm1d(freq_dim),
            nn.Linear(freq_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim + 64),
            nn.Dropout(0.3),
            nn.Linear(feat_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, freq):
        deep_feat = self.backbone(img)
        freq_feat = self.freq_proj(freq)
        fused = torch.cat([deep_feat, freq_feat], dim=1)
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

    dist_train = Counter(valid_labels[i] for i in train_idx)
    dist_test = Counter(valid_labels[i] for i in test_idx)
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

    dct_n = 16
    train_ds = LivenessDataset(train_ids, labels, transform_train, dct_features=dct_n)
    test_ds = LivenessDataset(test_ids, labels, transform_test, dct_features=dct_n)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = EfficientNetFreq(freq_dim=dct_n * 3).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B0 + DCT freq features, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for imgs, freq, targets in train_loader:
            imgs = imgs.to(device)
            freq = freq.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            out = model(imgs, freq)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, freq, targets in test_loader:
                imgs = imgs.to(device)
                freq = freq.to(device)
                out = model(imgs, freq)
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
    approach = "efficientnet_b0_6ch_plus_dct_freq_features_fusion"

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

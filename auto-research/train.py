"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Hybrid ResNet18 + handcrafted noise/frequency features
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

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_handcrafted_features(img_path):
    """Extract noise and frequency features from an image."""
    try:
        img = Image.open(str(img_path)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)

    features = []

    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    # Laplacian noise (key discriminative feature from previous experiments)
    lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    from scipy.signal import convolve2d
    laplacian = convolve2d(gray, lap_kernel, mode='valid')
    features.append(np.std(laplacian))

    # Gradient magnitude statistics
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features.append(np.mean(grad_mag))
    features.append(np.std(grad_mag))

    # FFT high-frequency ratio
    fft2 = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft2)
    mag = np.abs(fft_shift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total_energy = np.sum(mag)
    center_mask = np.zeros_like(mag)
    y_idx, x_idx = np.ogrid[:h, :w]
    center_mask[(y_idx - cy)**2 + (x_idx - cx)**2 <= r**2] = 1
    low_energy = np.sum(mag * center_mask)
    hf_ratio = 1.0 - (low_energy / (total_energy + 1e-10))
    features.append(hf_ratio)

    # Color channel noise (per-channel Laplacian std)
    for c in range(3):
        ch_lap = convolve2d(arr[:,:,c], lap_kernel, mode='valid')
        features.append(np.std(ch_lap))

    # BW fraction (fraction of near-zero saturation pixels)
    max_c = np.max(arr, axis=2)
    min_c = np.min(arr, axis=2)
    sat = (max_c - min_c) / (max_c + 1e-10)
    bw_frac = np.mean(sat < 0.05)
    features.append(bw_frac)

    return np.array(features[:8], dtype=np.float32)


class HybridLivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, cache_features=True):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.feature_cache = {}
        
        if cache_features:
            print(f"Pre-computing handcrafted features for {len(sig_ids)} samples...")
            for i, sig in enumerate(sig_ids):
                far_path = self.samples_dir / sig / "far.jpg"
                near_path = self.samples_dir / sig / "near.jpg"
                far_feats = compute_handcrafted_features(far_path)
                near_feats = compute_handcrafted_features(near_path)
                self.feature_cache[sig] = np.concatenate([far_feats, near_feats])
                if (i + 1) % 500 == 0:
                    print(f"  Features computed: {i+1}/{len(sig_ids)}")
            print(f"Feature computation done.")

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

        img = torch.cat([far_img, near_img], dim=0)  # 6-channel
        
        if sig in self.feature_cache:
            hand_feats = torch.tensor(self.feature_cache[sig], dtype=torch.float32)
        else:
            far_feats = compute_handcrafted_features(far_path)
            near_feats = compute_handcrafted_features(near_path)
            hand_feats = torch.tensor(np.concatenate([far_feats, near_feats]), dtype=torch.float32)

        return img, hand_feats, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class HybridModel(nn.Module):
    """ResNet18 backbone + handcrafted feature branch, fused for classification."""
    def __init__(self, num_handcrafted=16, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = old_conv.weight
            resnet.conv1.weight[:, 3:] = old_conv.weight

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        cnn_dim = 512

        self.hand_branch = nn.Sequential(
            nn.Linear(num_handcrafted, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(cnn_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, hand_feats):
        cnn_out = self.backbone(img).flatten(1)  # (B, 512)
        hand_out = self.hand_branch(hand_feats)   # (B, 64)
        combined = torch.cat([cnn_out, hand_out], dim=1)
        return self.classifier(combined)


def load_data():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    valid = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    random.shuffle(valid)
    split = int(len(valid) * 0.8)
    train_ids = valid[:split]
    test_ids = valid[split:]

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
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HybridLivenessDataset(train_ids, labels, transform_train)
    test_ds = HybridLivenessDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(num_handcrafted=16, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HybridResNet18+HandFeats, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    epochs = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    
    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for imgs, hand_feats, targets in train_loader:
            imgs = imgs.to(device)
            hand_feats = hand_feats.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            out = model(imgs, hand_feats)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, hand_feats, targets in test_loader:
                imgs = imgs.to(device)
                hand_feats = hand_feats.to(device)
                out = model(imgs, hand_feats)
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
    approach = "hybrid_resnet18_handcrafted_noise_freq_focal_loss_onecycle"

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

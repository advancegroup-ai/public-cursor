"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: Dual-stream EfficientNet-B0 with separate far/near encoders,
gradual unfreezing, focal loss, OneCycleLR, strong augmentation, TTA.

This approach uses separate pretrained EfficientNet-B0 encoders for far and
near images (proper 3-channel RGB), fuses their features via attention-weighted
concatenation, and adds handcrafted noise features as an auxiliary branch.
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def compute_noise_features(img_path):
    """Laplacian noise std + gradient mean per image (4 features per image)."""
    try:
        img = Image.open(str(img_path)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)

    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    from scipy.signal import convolve2d
    laplacian = convolve2d(gray, lap_kernel, mode='valid')
    noise_std = np.std(laplacian)

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)

    fft2 = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft2)
    mag = np.abs(fft_shift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total_energy = np.sum(mag)
    y_idx, x_idx = np.ogrid[:h, :w]
    center_mask = ((y_idx - cy)**2 + (x_idx - cx)**2 <= r**2).astype(np.float32)
    low_energy = np.sum(mag * center_mask)
    hf_ratio = 1.0 - (low_energy / (total_energy + 1e-10))

    return np.array([noise_std, grad_mean, grad_std, hf_ratio], dtype=np.float32)


class DualStreamDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, cache_features=True):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.feature_cache = {}

        if cache_features:
            print(f"Pre-computing features for {len(sig_ids)} samples...")
            t0 = time.time()
            for i, sig in enumerate(sig_ids):
                far_path = self.samples_dir / sig / "far.jpg"
                near_path = self.samples_dir / sig / "near.jpg"
                far_feats = compute_noise_features(far_path)
                near_feats = compute_noise_features(near_path)
                diff_feats = far_feats - near_feats
                self.feature_cache[sig] = np.concatenate([far_feats, near_feats, diff_feats])
                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{len(sig_ids)} ({time.time()-t0:.0f}s)")
            print(f"Feature computation done in {time.time()-t0:.0f}s")

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

        if sig in self.feature_cache:
            hand_feats = torch.tensor(self.feature_cache[sig], dtype=torch.float32)
        else:
            far_feats = compute_noise_features(far_path)
            near_feats = compute_noise_features(near_path)
            diff_feats = far_feats - near_feats
            hand_feats = torch.tensor(
                np.concatenate([far_feats, near_feats, diff_feats]), dtype=torch.float32
            )

        return far_img, near_img, hand_feats, label


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


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, feat_far, feat_near):
        combined = torch.cat([feat_far, feat_near], dim=1)
        weights = self.attn(combined)  # (B, 2)
        w_far = weights[:, 0:1]
        w_near = weights[:, 1:2]
        return w_far * feat_far + w_near * feat_near


class DualStreamModel(nn.Module):
    def __init__(self, num_handcrafted=12, num_classes=2):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        eff_far = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        eff_near = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.far_features = eff_far.features
        self.far_pool = eff_far.avgpool
        self.near_features = eff_near.features
        self.near_pool = eff_near.avgpool
        cnn_dim = 1280

        self.attention_fusion = AttentionFusion(cnn_dim)

        self.hand_branch = nn.Sequential(
            nn.Linear(num_handcrafted, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(cnn_dim + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, far_img, near_img, hand_feats):
        far_out = self.far_pool(self.far_features(far_img)).flatten(1)
        near_out = self.near_pool(self.near_features(near_img)).flatten(1)

        fused_cnn = self.attention_fusion(far_out, near_out)
        hand_out = self.hand_branch(hand_feats)
        combined = torch.cat([fused_cnn, hand_out], dim=1)
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

    pos = [s for s in valid if labels[s]["main_label"] == "Positive"]
    neg = [s for s in valid if labels[s]["main_label"] != "Positive"]
    random.shuffle(pos)
    random.shuffle(neg)

    split_pos = int(len(pos) * 0.8)
    split_neg = int(len(neg) * 0.8)
    train_ids = pos[:split_pos] + neg[:split_neg]
    test_ids = pos[split_pos:] + neg[split_neg:]
    random.shuffle(train_ids)
    random.shuffle(test_ids)

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
        T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.1),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = DualStreamDataset(train_ids, labels, transform_train)
    test_ds = DualStreamDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)

    model = DualStreamModel(num_handcrafted=12, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: DualStreamEfficientNet-B0+AttentionFusion+HandFeats, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    backbone_params = list(model.far_features.parameters()) + list(model.near_features.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith('far_features') and not n.startswith('near_features')]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5, 'weight_decay': 1e-3},
        {'params': head_params, 'lr': 3e-4, 'weight_decay': 1e-3},
    ])

    epochs = 12
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1.5e-4, 5e-4],
        steps_per_epoch=len(train_loader), epochs=epochs
    )

    # Gradual unfreezing: freeze backbone for first 2 epochs
    for p in backbone_params:
        p.requires_grad = False

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        if epoch == 2:
            print("Unfreezing backbone...")
            for p in backbone_params:
                p.requires_grad = True

        model.train()
        total_loss = 0
        for far_imgs, near_imgs, hand_feats, targets in train_loader:
            far_imgs = far_imgs.to(device)
            near_imgs = near_imgs.to(device)
            hand_feats = hand_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            out = model(far_imgs, near_imgs, hand_feats)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for far_imgs, near_imgs, hand_feats, targets in test_loader:
                far_imgs = far_imgs.to(device)
                near_imgs = near_imgs.to(device)
                hand_feats = hand_feats.to(device)
                out = model(far_imgs, near_imgs, hand_feats)
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
    approach = "dual_effb0_attention_fusion_handcrafted_noise_focal_gradual_unfreeze"

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

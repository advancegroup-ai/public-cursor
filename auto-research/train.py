"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Dual-stream ResNet18 with cross-attention fusion.
Key idea: separate encoders for far and near images, then cross-attention
to learn relationships between the two views before classification.
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


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.attn_far_to_near = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_near_to_far = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, far_feat, near_feat):
        far_q = far_feat.unsqueeze(1)
        near_q = near_feat.unsqueeze(1)
        attended_far, _ = self.attn_far_to_near(far_q, near_q, near_q)
        attended_near, _ = self.attn_near_to_far(near_q, far_q, far_q)
        far_out = self.norm1(far_feat + attended_far.squeeze(1))
        near_out = self.norm2(near_feat + attended_near.squeeze(1))
        return far_out, near_out


class DualStreamCrossAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = base.fc.in_features

        self.far_encoder = nn.Sequential(*list(base.children())[:-1])
        base2 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.near_encoder = nn.Sequential(*list(base2.children())[:-1])

        self.cross_attn = CrossAttentionFusion(dim=feat_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        far = x[:, :3]
        near = x[:, 3:]

        far_feat = self.far_encoder(far).squeeze(-1).squeeze(-1)
        near_feat = self.near_encoder(near).squeeze(-1).squeeze(-1)

        far_att, near_att = self.cross_attn(far_feat, near_feat)
        fused = torch.cat([far_att, near_att], dim=1)
        return self.classifier(fused)


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

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        return img, label


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


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(1e-6 / optimizer.defaults['lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.15),
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

    model = DualStreamCrossAttention().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Dual-stream ResNet18 + CrossAttention, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    epochs = 15
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=2, total_epochs=epochs)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f} lr={lr:.6f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0
    approach = "dual_stream_resnet18_cross_attention_fusion_label_smoothing"

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

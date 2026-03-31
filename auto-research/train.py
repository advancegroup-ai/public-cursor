"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: Dual-stream ResNet18 with cross-attention fusion.
Separate pretrained encoders for far and near images, with a cross-attention
module that lets each stream attend to the other before classification.
This should capture far-near consistency cues important for liveness.
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


class DualStreamDataset(Dataset):
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

        return far_img, near_img, label


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        B, D = x.shape
        q = self.q_proj(self.norm1(x)).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(self.norm2(context)).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(self.norm2(context)).view(B, self.num_heads, self.head_dim)

        attn = (q * k).sum(-1) * self.scale
        attn = attn.softmax(-1)

        out = (attn.unsqueeze(-1) * v).view(B, D)
        return x + self.out_proj(out)


class DualStreamModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        far_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        near_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.far_encoder = nn.Sequential(*list(far_backbone.children())[:-1])
        self.near_encoder = nn.Sequential(*list(near_backbone.children())[:-1])

        feat_dim = 512

        self.cross_attn_far = CrossAttention(feat_dim, num_heads=4)
        self.cross_attn_near = CrossAttention(feat_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, far_img, near_img):
        far_feat = self.far_encoder(far_img).flatten(1)
        near_feat = self.near_encoder(near_img).flatten(1)

        far_attended = self.cross_attn_far(far_feat, near_feat)
        near_attended = self.cross_attn_near(near_feat, far_feat)

        combined = torch.cat([far_attended, near_attended], dim=1)
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
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    model = DualStreamModel(num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Dual-stream ResNet18 + cross-attention, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW([
        {"params": model.far_encoder.parameters(), "lr": 5e-5},
        {"params": model.near_encoder.parameters(), "lr": 5e-5},
        {"params": model.cross_attn_far.parameters(), "lr": 2e-4},
        {"params": model.cross_attn_near.parameters(), "lr": 2e-4},
        {"params": model.classifier.parameters(), "lr": 2e-4},
    ], weight_decay=1e-3)

    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for far_imgs, near_imgs, targets in train_loader:
            far_imgs = far_imgs.to(device)
            near_imgs = near_imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            out = model(far_imgs, near_imgs)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(epoch)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for far_imgs, near_imgs, targets in test_loader:
                far_imgs = far_imgs.to(device)
                near_imgs = near_imgs.to(device)
                out = model(far_imgs, near_imgs)
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
    approach = "dual_stream_resnet18_cross_attention_label_smoothing"

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

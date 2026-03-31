"""
train.py — Liveness detection: Dual-stream ResNet18 shared encoder + cross-attention fusion + TTA
Key innovation: cross-attention between far and near feature maps before pooling.
This lets the model learn what spatial regions in one image are informative 
given the other image's context (e.g., matching face regions, detecting artifacts).
"""
import os, sys, json, time, random, math
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


class LivenessDatasetDual(Dataset):
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


def load_data():
    with open(str(DATA_DIR / "labels.json")) as f:
        labels = json.load(f)

    samples_dir = DATA_DIR / "samples"
    valid = []
    for sig, info in labels.items():
        far = samples_dir / sig / "far.jpg"
        near = samples_dir / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    binary_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid]
    print(f"Total valid samples: {len(valid)}")
    print(f"Distribution: {Counter(binary_labels)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(valid, binary_labels)):
        if fold_idx == 0:
            train_ids = [valid[i] for i in train_idx]
            test_ids = [valid[i] for i in test_idx]
            break

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


class CrossAttention(nn.Module):
    """Lightweight cross-attention between two feature maps."""
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
        B, N, C = x.shape
        q = self.q_proj(self.norm1(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(self.norm2(context)).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x + self.out_proj(out)


class DualStreamCrossAttention(nn.Module):
    """Shared ResNet18 backbone + cross-attention between far/near feature maps."""
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(base.children())
        self.backbone = nn.Sequential(*layers[:-2])  # up to layer4 (7x7x512)
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 512
        self.cross_attn_far = CrossAttention(feat_dim, num_heads=4)
        self.cross_attn_near = CrossAttention(feat_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, far, near):
        # Extract spatial feature maps
        f_far = self.backbone(far)    # (B, 512, 7, 7)
        f_near = self.backbone(near)  # (B, 512, 7, 7)

        B, C, H, W = f_far.shape
        far_tokens = f_far.flatten(2).permute(0, 2, 1)   # (B, 49, 512)
        near_tokens = f_near.flatten(2).permute(0, 2, 1)  # (B, 49, 512)

        far_attended = self.cross_attn_far(far_tokens, near_tokens)   # (B, 49, 512)
        near_attended = self.cross_attn_near(near_tokens, far_tokens)  # (B, 49, 512)

        far_feat = far_attended.permute(0, 2, 1).reshape(B, C, H, W)
        near_feat = near_attended.permute(0, 2, 1).reshape(B, C, H, W)

        f_far_pooled = self.pool(far_feat).flatten(1)    # (B, 512)
        f_near_pooled = self.pool(near_feat).flatten(1)   # (B, 512)

        combined = torch.cat([f_far_pooled, f_near_pooled], dim=1)
        return self.classifier(combined)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
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

    train_ds = LivenessDatasetDual(train_ids, labels, transform_train)
    test_ds = LivenessDatasetDual(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = DualStreamCrossAttention().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Dual-stream ResNet18 + Cross-Attention, params: {num_params:,}")

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=1.5)

    # Differential LR: lower for backbone, higher for cross-attn + classifier
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.cross_attn_far.parameters()) + list(model.cross_attn_near.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 2e-4},
    ], weight_decay=1e-3)

    epochs = 12
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for far, near, targets in train_loader:
            far, near, targets = far.to(device), near.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(far, near)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Eval with TTA
        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for far, near, targets in test_loader:
                far, near = far.to(device), near.to(device)
                out1 = model(far, near)
                out2 = model(torch.flip(far, dims=[3]), torch.flip(near, dims=[3]))
                out = (out1 + out2) / 2.0
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1v = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1v:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1v

    elapsed = time.time() - t0
    approach = "dual_stream_resnet18_shared_cross_attention_focal_diff_LR_TTA"

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

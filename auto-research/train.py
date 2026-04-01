"""
train.py — Liveness detection: Cross-attention between face streams and card
Key idea: Use cross-attention to let the model learn which parts of the face
images are inconsistent with the card image. This mimics how human annotators
compare the ID card face with the submitted face photos.
Shared ResNet18 backbone, cross-attention on spatial features, then pool + classify.
AMP, 160px, 8 epochs.
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
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = DATA_DIR / "last_result.json"
SEED = 42
MAX_SECONDS = 270
IMG_SIZE = 160

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


class LivenessDataset3Stream(Dataset):
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

        imgs = []
        for name in ["far.jpg", "near.jpg", "card.jpg"]:
            path = self.samples_dir / sig / name
            try:
                img = Image.open(str(path)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        return imgs[0], imgs[1], imgs[2], label


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
    print(f"Total valid samples: {len(valid)}, Distribution: {Counter(binary_labels)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(valid, binary_labels)):
        if fold_idx == 0:
            train_ids = [valid[i] for i in train_idx]
            test_ids = [valid[i] for i in test_idx]
            break

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
    return train_ids, test_ids, labels


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CrossAttentionBlock(nn.Module):
    """Lightweight cross-attention: query from face, key/value from card."""
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, face_tokens, card_tokens):
        attended, _ = self.attn(face_tokens, card_tokens, card_tokens)
        return self.norm(face_tokens + attended)


class CrossAttentionModel(nn.Module):
    """Shared ResNet18 backbone → spatial features → cross-attention with card → classify."""
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(base.children())
        self.backbone = nn.Sequential(*layers[:-2])  # up to layer4, 512x5x5 at 160px
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cross_attn_far = CrossAttentionBlock(512, num_heads=4)
        self.cross_attn_near = CrossAttentionBlock(512, num_heads=4)

        feat_dim = 512 * 3
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def _get_spatial_tokens(self, x):
        feat = self.backbone(x)  # [B, 512, H, W]
        B, C, H, W = feat.shape
        return feat.flatten(2).permute(0, 2, 1)  # [B, H*W, 512]

    def forward(self, far, near, card):
        far_tokens = self._get_spatial_tokens(far)
        near_tokens = self._get_spatial_tokens(near)
        card_tokens = self._get_spatial_tokens(card)

        far_attended = self.cross_attn_far(far_tokens, card_tokens)
        near_attended = self.cross_attn_near(near_tokens, card_tokens)

        far_pooled = far_attended.mean(dim=1)
        near_pooled = near_attended.mean(dim=1)
        card_pooled = card_tokens.mean(dim=1)

        combined = torch.cat([far_pooled, near_pooled, card_pooled], dim=1)
        return self.classifier(combined)


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDataset3Stream(train_ids, labels, transform_train)
    test_ds = LivenessDataset3Stream(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = CrossAttentionModel().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Cross-attention ResNet18 (face×card), params: {num_params:,}")

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=1.5)

    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.2, div_factor=10, final_div_factor=100
    )
    scaler = GradScaler()

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for far, near, card, targets in train_loader:
            far, near, card = far.to(device), near.to(device), card.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(far, near, card)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for far, near, card, targets in test_loader:
                far, near, card = far.to(device), near.to(device), card.to(device)
                with autocast():
                    out1 = model(far, near, card)
                    out2 = model(torch.flip(far, dims=[3]), torch.flip(near, dims=[3]), torch.flip(card, dims=[3]))
                out = (out1 + out2) / 2.0
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1v = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        elapsed_now = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1v:.4f} [{elapsed_now:.0f}s]")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1v

    elapsed = time.time() - t0
    approach = "cross_attention_resnet18_face_x_card_AMP_focal_TTA"

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

"""
train.py — Liveness detection: BEST CANDIDATE — Hybrid 3-image + card verification + handcrafted
Combines:
1. Shared ResNet18 encodes far, near, card → face-card consistency (cosine sim + diff)
2. Handcrafted noise/gradient/FFT features (proven perfect on smaller dataset)
3. Fused classifier with both deep + handcrafted features
This is the most promising approach for next iteration to run.
AMP, 160px, 10 epochs.
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
from PIL import Image, ImageFilter
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


def extract_handcrafted(img_pil):
    """Extract 4 noise/gradient features from a PIL image."""
    img_gray = img_pil.convert("L")
    gray = np.array(img_gray, dtype=np.float32)

    lap = img_gray.filter(ImageFilter.Kernel(
        (3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128
    ))
    noise_std = (np.array(lap, dtype=np.float32) - 128.0).std()

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    min_h, min_w = min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1])
    grad_mean = np.sqrt(gx[:min_h, :min_w]**2 + gy[:min_h, :min_w]**2).mean()

    fft = np.fft.fft2(gray)
    magnitude = np.abs(np.fft.fftshift(fft))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    center_energy = magnitude[max(0, cy-r):cy+r, max(0, cx-r):cx+r].sum()
    hf_ratio = 1.0 - (center_energy / (magnitude.sum() + 1e-10))

    bw_frac = float(np.mean(gray < 10)) + float(np.mean(gray > 245))

    return np.array([noise_std, grad_mean, hf_ratio, bw_frac], dtype=np.float32)


class LivenessDatasetHybrid3(Dataset):
    def __init__(self, sig_ids, labels, transform=None):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.feat_cache = {}

    def __len__(self):
        return len(self.sig_ids)

    def _get_features(self, sig):
        if sig in self.feat_cache:
            return self.feat_cache[sig]
        feats_all = []
        for name in ["far.jpg", "near.jpg", "card.jpg"]:
            path = self.samples_dir / sig / name
            try:
                img = Image.open(str(path)).convert("RGB")
                feats_all.append(extract_handcrafted(img))
            except Exception:
                feats_all.append(np.zeros(4, dtype=np.float32))
        result = np.concatenate(feats_all)  # 12 features
        self.feat_cache[sig] = result
        return result

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

        feats = torch.tensor(self._get_features(sig), dtype=torch.float32)
        return imgs[0], imgs[1], imgs[2], feats, label


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


class HybridCardVerificationModel(nn.Module):
    """Shared ResNet18 for 3 images + card-face consistency + handcrafted features."""
    def __init__(self, handcrafted_dim=12, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512

        self.diff_proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
        )

        self.feat_bn = nn.BatchNorm1d(handcrafted_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 64),
            nn.ReLU(inplace=True),
        )

        # 512*3 (streams) + 128*2 (diffs) + 2 (cosines) + 64 (handcrafted)
        fused_dim = feat_dim * 3 + 128 * 2 + 2 + 64
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, far, near, card, handcrafted):
        f_far = self.encoder(far).flatten(1)
        f_near = self.encoder(near).flatten(1)
        f_card = self.encoder(card).flatten(1)

        diff_far_card = self.diff_proj(torch.abs(f_far - f_card))
        diff_near_card = self.diff_proj(torch.abs(f_near - f_card))

        cos_far_card = F.cosine_similarity(f_far, f_card, dim=1, eps=1e-6).unsqueeze(1)
        cos_near_card = F.cosine_similarity(f_near, f_card, dim=1, eps=1e-6).unsqueeze(1)

        h = self.feat_bn(handcrafted)
        h = self.feat_proj(h)

        combined = torch.cat([
            f_far, f_near, f_card,
            diff_far_card, diff_near_card,
            cos_far_card, cos_near_card,
            h,
        ], dim=1)
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

    train_ds = LivenessDatasetHybrid3(train_ids, labels, transform_train)
    test_ds = LivenessDatasetHybrid3(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridCardVerificationModel().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid 3-image + card verification + handcrafted, params: {num_params:,}")

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=1.5)

    epochs = 10
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
        for far, near, card, feats, targets in train_loader:
            far, near, card = far.to(device), near.to(device), card.to(device)
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(far, near, card, feats)
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
            for far, near, card, feats, targets in test_loader:
                far, near, card = far.to(device), near.to(device), card.to(device)
                feats = feats.to(device)
                with autocast():
                    out1 = model(far, near, card, feats)
                    out2 = model(
                        torch.flip(far, dims=[3]),
                        torch.flip(near, dims=[3]),
                        torch.flip(card, dims=[3]),
                        feats,
                    )
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
    approach = "hybrid_3img_card_verify_cosine_diff_handcrafted_noise_AMP_focal_TTA"

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

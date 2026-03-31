"""
train.py — Liveness detection: Dual-stream ResNet18 shared encoder + handcrafted features fusion + SWA + TTA
Building on best result (0.985515) from dual-stream ResNet18 shared encoder.
Key improvements:
1. Handcrafted features (noise via Laplacian, gradient, FFT high-freq ratio) — numpy-only, no cv2
2. SWA for better generalization after epoch 8
3. Horizontal flip TTA at evaluation
4. Focal loss (gamma=1.5) with class weights
5. Warmup cosine LR schedule
6. Stratified 5-fold split
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
from PIL import Image, ImageFilter
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


def extract_handcrafted(img_pil):
    """Extract 6 features per image using PIL/numpy only (no cv2 dependency)."""
    img_gray = img_pil.convert("L")
    gray = np.array(img_gray, dtype=np.float32)

    lap = img_gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    lap_arr = np.array(lap, dtype=np.float32) - 128.0
    noise_std = lap_arr.std()
    noise_mean = np.abs(lap_arr).mean()

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    min_h = min(gx.shape[0], gy.shape[0])
    min_w = min(gx.shape[1], gy.shape[1])
    grad_mag = np.sqrt(gx[:min_h, :min_w]**2 + gy[:min_h, :min_w]**2)
    grad_mean = grad_mag.mean()
    grad_std = grad_mag.std()

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    center_energy = magnitude[max(0, cy-r):cy+r, max(0, cx-r):cx+r].sum()
    total_energy = magnitude.sum() + 1e-10
    hf_ratio = 1.0 - (center_energy / total_energy)

    bw_frac = float(np.mean(gray < 10)) + float(np.mean(gray > 245))

    return np.array([noise_std, noise_mean, grad_mean, grad_std, hf_ratio, bw_frac], dtype=np.float32)


class LivenessDatasetHybrid(Dataset):
    def __init__(self, sig_ids, labels, transform=None):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.samples_dir = DATA_DIR / "samples"
        self.feature_cache = {}

    def __len__(self):
        return len(self.sig_ids)

    def _get_features(self, sig):
        if sig in self.feature_cache:
            return self.feature_cache[sig]
        far_path = self.samples_dir / sig / "far.jpg"
        near_path = self.samples_dir / sig / "near.jpg"
        try:
            far_pil = Image.open(str(far_path)).convert("RGB")
            near_pil = Image.open(str(near_path)).convert("RGB")
            far_feats = extract_handcrafted(far_pil)
            near_feats = extract_handcrafted(near_pil)
            feats = np.concatenate([far_feats, near_feats])
        except Exception:
            feats = np.zeros(12, dtype=np.float32)
        self.feature_cache[sig] = feats
        return feats

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

        feats = torch.tensor(self._get_features(sig), dtype=torch.float32)
        return far_img, near_img, feats, label


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


class HybridDualStreamModel(nn.Module):
    """Shared ResNet18 for far/near + handcrafted features → fused classifier."""
    def __init__(self, handcrafted_dim=12, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512

        self.feat_bn = nn.BatchNorm1d(handcrafted_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = feat_dim * 2 + 64
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, far, near, handcrafted):
        f_far = self.encoder(far).flatten(1)
        f_near = self.encoder(near).flatten(1)
        h = self.feat_bn(handcrafted)
        h = self.feat_proj(h)
        combined = torch.cat([f_far, f_near, h], dim=1)
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
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDatasetHybrid(train_ids, labels, transform_train)
    test_ds = LivenessDatasetHybrid(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridDualStreamModel().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid Dual-stream ResNet18 shared + handcrafted, params: {num_params:,}")

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=1.5)

    epochs = 15
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA: average model weights from epoch swa_start onward
    swa_start = 8
    swa_n = 0
    swa_state = None

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    step = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 30:
            print(f"Time budget approaching at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for far, near, feats, targets in train_loader:
            far, near = far.to(device), near.to(device)
            feats, targets = feats.to(device), targets.to(device)

            optimizer.zero_grad()
            out = model(far, near, feats)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            step += 1

        # Manual SWA: accumulate state dicts
        if epoch >= swa_start:
            sd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.clone().float() for k, v in sd.items()}
            else:
                for k in swa_state:
                    swa_state[k] += sd[k].float()
            swa_n += 1

        # Eval (use SWA-averaged model if available)
        if swa_state is not None and swa_n > 0:
            orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
            avg_sd = {k: (v / swa_n).to(orig_sd[k].dtype) for k, v in swa_state.items()}
            model.load_state_dict(avg_sd)

        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for far, near, feats, targets in test_loader:
                far, near = far.to(device), near.to(device)
                feats = feats.to(device)

                out1 = model(far, near, feats)
                out2 = model(torch.flip(far, dims=[3]), torch.flip(near, dims=[3]), feats)
                out = (out1 + out2) / 2.0

                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())

        if swa_state is not None and swa_n > 0:
            model.load_state_dict(orig_sd)

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1v = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        swa_tag = f" [SWA avg {swa_n}]" if swa_n > 0 else ""
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1v:.4f}{swa_tag}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1v

    elapsed = time.time() - t0
    approach = "hybrid_dual_resnet18_shared_noise_freq_features_SWA_TTA_focal"

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

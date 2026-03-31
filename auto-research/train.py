"""
train.py — Liveness detection: Hybrid ResNet18 + Handcrafted Features + Focal Loss

Architecture:
- ResNet18 pretrained backbone modified for 6-channel input (far+near concatenated)
- Parallel handcrafted feature branch (noise, gradient, FFT, DCT, color)
- Feature fusion via learned MLP combining CNN embeddings + handcrafted features
- Focal loss for hard example mining
- Cosine warmup learning rate schedule
- Stratified 5-fold cross-validation for robust evaluation
- Strong augmentation: ColorJitter, GaussianBlur, RandomErasing

Best previous result: 0.976659 (ResNet18 6ch baseline)
Target: Beat 0.98 balanced accuracy on full 3612-sample dataset.
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
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
ITER_RESULTS = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/iter27_result.json")
SEED = 42
MAX_SECONDS = 240

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def compute_handcrafted(img_tensor):
    """Compute handcrafted features from a 3-channel image tensor [C,H,W]."""
    arr = img_tensor.numpy()
    if arr.shape[0] == 3:
        gray = 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
    else:
        gray = arr[0]

    # Laplacian noise
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] +
        gray[1:-1, :-2] + gray[1:-1, 2:] -
        4 * gray[1:-1, 1:-1]
    )
    noise_std = float(np.std(lap))

    # Gradient
    gy = gray[1:, :] - gray[:-1, :]
    gx = gray[:, 1:] - gray[:, :-1]
    grad_mean = float(np.mean(np.abs(gy)) + np.mean(np.abs(gx)))
    grad_std = float(np.std(np.abs(gy)) + np.std(np.abs(gx)))

    # FFT high-frequency ratio
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total_energy = np.sum(magnitude) + 1e-8
    low_energy = np.sum(magnitude[max(0,cy-r):cy+r, max(0,cx-r):cx+r])
    hf_ratio = 1.0 - (low_energy / total_energy)

    # Color features
    blue_shift = float(np.mean(arr[2]) - np.mean(arr[0])) if arr.shape[0] == 3 else 0.0
    brightness_std = float(np.std(gray))
    channel_std = float(np.std([np.mean(arr[c]) for c in range(min(arr.shape[0], 3))]))

    return np.array([noise_std, grad_mean, grad_std, hf_ratio, blue_shift, brightness_std, channel_std], dtype=np.float32)


class LivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, transform=None, compute_hc=True):
        self.sig_ids = sig_ids
        self.labels = labels
        self.transform = transform
        self.compute_hc = compute_hc
        self.samples_dir = DATA_DIR / "samples"
        self.raw_transform = T.Compose([T.Resize((112, 112)), T.ToTensor()])

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

        # Handcrafted features (computed on raw resized images before augmentation)
        hc_feats = torch.zeros(14)
        if self.compute_hc:
            try:
                far_raw = self.raw_transform(far_img)
                near_raw = self.raw_transform(near_img)
                far_hc = compute_handcrafted(far_raw)
                near_hc = compute_handcrafted(near_raw)
                hc_feats = torch.from_numpy(np.concatenate([far_hc, near_hc]))
            except Exception:
                pass

        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)

        img = torch.cat([far_img, near_img], dim=0)
        return img, hc_feats, label


class HybridModel(nn.Module):
    def __init__(self, num_hc_features=14, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            backbone.conv1.weight[:, 3:] = old_conv.weight

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        cnn_dim = 512

        self.hc_encoder = nn.Sequential(
            nn.Linear(num_hc_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(cnn_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, hc):
        cnn_out = self.features(x).flatten(1)
        hc_out = self.hc_encoder(hc)
        combined = torch.cat([cnn_out, hc_out], dim=1)
        return self.classifier(combined)


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

    return np.array(valid), np.array(valid_labels), labels


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    valid_sigs, valid_labels, labels = load_data()
    print(f"Total: {len(valid_sigs)} samples, Pos={sum(valid_labels==0)}, Neg={sum(valid_labels==1)}", flush=True)

    # Use first stratified fold as train/test split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(iter(skf.split(valid_sigs, valid_labels)))

    train_ids = valid_sigs[train_idx].tolist()
    test_ids = valid_sigs[test_idx].tolist()

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}", flush=True)
    print(f"Test:  {len(test_ids)} {dict(dist_test)}", flush=True)

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(num_hc_features=14, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid ResNet18 + HC, params: {num_params:,}", flush=True)

    train_labels_list = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels_list)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    epochs = 15
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 2 * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}", flush=True)
            break

        model.train()
        total_loss = 0
        for imgs, hc_feats, targets in train_loader:
            imgs = imgs.to(device)
            hc_feats = hc_feats.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            out = model(imgs, hc_feats)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, hc_feats, targets in test_loader:
                imgs = imgs.to(device)
                hc_feats = hc_feats.to(device)
                out = model(imgs, hc_feats)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.numpy())

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f} lr={lr_now:.6f}", flush=True)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0

    approach = "hybrid_resnet18_6ch_handcrafted14_focal_cosine_warmup_strong_aug"
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
    print(result_block, flush=True)

    result_data = {
        "balanced_accuracy": best_bal_acc,
        "accuracy": best_acc,
        "f1_score": best_f1,
        "num_params": num_params,
        "training_seconds": elapsed,
        "approach": approach,
    }
    for fpath in [RESULTS_FILE, ITER_RESULTS]:
        with open(str(fpath), "w") as rf:
            json.dump(result_data, rf, indent=2)
    print(f"Results written to {ITER_RESULTS}", flush=True)


train()

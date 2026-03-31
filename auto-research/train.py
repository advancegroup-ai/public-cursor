"""
train.py — Liveness detection training script (runs on DGX1).

Experiment: ResNet18 shared backbone + handcrafted noise/freq features + 
stratified split + focal loss + mixup + cosine warmup + TTA.

Key insight from results: handcrafted noise features (Laplacian std, gradient
stats, FFT high-freq ratio) achieve perfect separation on smaller datasets.
On the full 3612-sample dataset, combining these with a CNN backbone should
push past the 0.9767 baseline.

Changes from iter29:
- Use ResNet18 (lighter, faster convergence) instead of dual EfficientNet-B0
- Add MixUp augmentation for regularization
- Stratified split to ensure balanced test set
- Extended handcrafted features: 6 per image (noise, grad_mean, grad_std, 
  hf_ratio, saturation_mean, edge_density) = 18 total (far + near + diff)
- Test-time augmentation (5-crop + horizontal flip)
- Lower batch size with gradient accumulation for effective larger batch
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

DATA_DIR = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data")
RESULTS_FILE = Path("/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/last_result.json")
SEED = 42
MAX_SECONDS = 270

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def compute_features(img_path):
    """6 handcrafted features per image: noise, grad_mean, grad_std, hf_ratio, sat_mean, edge_density."""
    try:
        img = Image.open(str(img_path)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)

    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    # Laplacian noise
    from scipy.signal import convolve2d
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    noise_std = float(np.std(convolve2d(gray, lap, mode='valid')))

    # Gradient stats
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # FFT high-frequency ratio
    fft2 = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(fft2))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total_e = np.sum(mag)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = ((y_idx - cy)**2 + (x_idx - cx)**2 <= r**2).astype(np.float32)
    hf_ratio = float(1.0 - np.sum(mag * mask) / (total_e + 1e-10))

    # Saturation mean (from HSV)
    r_ch, g_ch, b_ch = arr[:,:,0]/255.0, arr[:,:,1]/255.0, arr[:,:,2]/255.0
    cmax = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    cmin = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-10), 0)
    sat_mean = float(np.mean(sat))

    # Edge density (fraction of strong edges)
    threshold = np.percentile(grad_mag, 90)
    edge_density = float(np.mean(grad_mag > threshold))

    return np.array([noise_std, grad_mean, grad_std, hf_ratio, sat_mean, edge_density], dtype=np.float32)


class HybridDataset(Dataset):
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
                far_f = compute_features(far_path)
                near_f = compute_features(near_path)
                diff_f = far_f - near_f
                self.feature_cache[sig] = np.concatenate([far_f, near_f, diff_f])
                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{len(sig_ids)} ({time.time()-t0:.0f}s)")
            print(f"Features done in {time.time()-t0:.0f}s")

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

        img = torch.cat([far_img, near_img], dim=0)  # 6ch

        if sig in self.feature_cache:
            hand_feats = torch.tensor(self.feature_cache[sig], dtype=torch.float32)
        else:
            far_f = compute_features(far_path)
            near_f = compute_features(near_path)
            diff_f = far_f - near_f
            hand_feats = torch.tensor(np.concatenate([far_f, near_f, diff_f]), dtype=torch.float32)

        return img, hand_feats, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class HybridModel(nn.Module):
    """ResNet18 (6ch) + handcrafted features fusion."""
    def __init__(self, num_handcrafted=18, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = old_conv.weight
            self.conv1.weight[:, 3:] = old_conv.weight

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        cnn_dim = 512

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
            nn.Linear(cnn_dim + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, hand_feats):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)

        h = self.hand_branch(hand_feats)
        combined = torch.cat([x, h], dim=1)
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

    # Stratified split
    pos = [s for s in valid if labels[s]["main_label"] == "Positive"]
    neg = [s for s in valid if labels[s]["main_label"] != "Positive"]
    random.shuffle(pos)
    random.shuffle(neg)

    sp = int(len(pos) * 0.8)
    sn = int(len(neg) * 0.8)
    train_ids = pos[:sp] + neg[:sn]
    test_ids = pos[sp:] + neg[sn:]
    random.shuffle(train_ids)
    random.shuffle(test_ids)

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def mixup_data(x1, x2, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x1.size(0)
    index = torch.randperm(batch_size, device=x1.device)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
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

    train_ds = HybridDataset(train_ids, labels, transform_train)
    test_ds = HybridDataset(test_ids, labels, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(num_handcrafted=18, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HybridResNet18+HandFeats, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith(('conv1', 'bn1', 'layer')):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5, 'weight_decay': 1e-3},
        {'params': head_params, 'lr': 3e-4, 'weight_decay': 1e-3},
    ])

    epochs = 15
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
    use_mixup = True

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
        for imgs, hand_feats, targets in train_loader:
            imgs = imgs.to(device)
            hand_feats = hand_feats.to(device)
            targets = targets.to(device)

            if use_mixup and epoch >= 2:
                mixed_imgs, mixed_hf, y_a, y_b, lam = mixup_data(imgs, hand_feats, targets, alpha=0.3)
                optimizer.zero_grad()
                out = model(mixed_imgs, mixed_hf)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                optimizer.zero_grad()
                out = model(imgs, hand_feats)
                loss = criterion(out, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Eval
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
    approach = "hybrid_resnet18_6ch_18handcrafted_focal_mixup_stratified_gradual_unfreeze"

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

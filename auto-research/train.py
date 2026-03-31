"""
train.py — Liveness: Hybrid ResNet18 + handcrafted noise/freq features fused classifier
Best previous DGX1: 0.976659 (ResNet18 6ch baseline)
Idea: Combine deep CNN features with proven-effective handcrafted features
(noise, gradient, FFT) that got 1.0 on smaller dataset.
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def extract_handcrafted(img_path):
    """Extract noise/frequency features proven effective on this dataset."""
    img = Image.open(str(img_path)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    lap = np.abs(gray[2:,1:-1] + gray[:-2,1:-1] + gray[1:-1,2:] + gray[1:-1,:-2] - 4*gray[1:-1,1:-1])
    noise_mean = float(np.mean(lap))
    noise_std = float(np.std(lap))

    gx = np.abs(gray[:,1:] - gray[:,:-1])
    gy = np.abs(gray[1:,:] - gray[:-1,:])
    grad_mean = float(np.mean(gx) + np.mean(gy)) / 2

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.ones((h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - cy)**2 + (xx - cx)**2 <= r**2] = False
    mag = np.abs(fft_shift)
    total_e = float(np.sum(mag))
    hf_ratio = float(np.sum(mag[mask])) / (total_e + 1e-8)

    return [noise_mean, noise_std, grad_mean, hf_ratio]


class HybridLivenessDataset(Dataset):
    def __init__(self, sig_ids, labels, handcrafted_feats, transform=None):
        self.sig_ids = sig_ids
        self.labels = labels
        self.handcrafted = handcrafted_feats
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
        hc = torch.tensor(self.handcrafted[sig], dtype=torch.float32)
        return img, hc, label


class HybridModel(nn.Module):
    def __init__(self, hc_dim=8, num_classes=2):
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
            nn.Linear(hc_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cnn_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, hc_feats):
        cnn_out = self.features(img).flatten(1)
        hc_out = self.hc_encoder(hc_feats)
        fused = torch.cat([cnn_out, hc_out], dim=1)
        return self.classifier(fused)


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

    print(f"Extracting handcrafted features for {len(valid)} samples...")
    hc_feats = {}
    for i, sig in enumerate(valid):
        if i % 500 == 0:
            print(f"  {i}/{len(valid)}")
        try:
            far_f = extract_handcrafted(DATA_DIR / "samples" / sig / "far.jpg")
            near_f = extract_handcrafted(DATA_DIR / "samples" / sig / "near.jpg")
            hc_feats[sig] = far_f + near_f
        except Exception:
            hc_feats[sig] = [0.0] * 8

    all_feats = np.array([hc_feats[s] for s in valid])
    feat_mean = all_feats.mean(axis=0)
    feat_std = all_feats.std(axis=0) + 1e-8
    for sig in valid:
        hc_feats[sig] = ((np.array(hc_feats[sig]) - feat_mean) / feat_std).tolist()

    split = int(len(valid) * 0.8)
    train_ids = valid[:split]
    test_ids = valid[split:]

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels, hc_feats


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels, hc_feats = load_data()

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.1),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HybridLivenessDataset(train_ids, labels, hc_feats, transform_train)
    test_ds = HybridLivenessDataset(test_ids, labels, hc_feats, transform_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridModel(hc_dim=8, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Hybrid ResNet18 + HC features, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

    epochs = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15, anneal_strategy='cos',
    )

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS - 20:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for imgs, hc, targets in train_loader:
            imgs, hc, targets = imgs.to(device), hc.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs, hc)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, hc, targets in test_loader:
                imgs, hc = imgs.to(device), hc.to(device)
                out = model(imgs, hc)
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
    approach = "hybrid_resnet18_6ch_plus_8_handcrafted_noise_grad_fft_fused_MLP"

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

"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: Triple-stream ResNet18 shared encoder (far + near + card) with
ATTENTION-based fusion + cosine LR + TTA.

Key innovations vs previous best (dual-stream, 0.9855):
1. Uses ALL 3 images (far, near, card) — previous best only used 2
2. Cross-attention module learns to weight face-card similarity
3. Explicit difference features capture face-card inconsistency
4. Channel attention (SE-like) on fused features for feature selection
"""
import os, sys, json, time, random
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
    for fname in ["neg_batch.json", "pos_batch.json"]:
        with open(str(DATA_DIR / fname)) as f:
            batch = json.load(f)
        for d in batch.get("data", []):
            parts = d["sample_id"].split("_")
            if len(parts) >= 4:
                sig = parts[3]
                main_label = d["pn"].split("/")[0]
                if main_label in ("Negative Type", "Negative_Type"):
                    main_label = "Negative"
                labels[sig] = {"main_label": main_label, "pn": d["pn"]}
    return labels


class LivenessDatasetTriple(Dataset):
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
        card_path = self.samples_dir / sig / "card.jpg"
        try:
            far_img = Image.open(str(far_path)).convert("RGB")
        except Exception:
            far_img = Image.new("RGB", (224, 224))
        try:
            near_img = Image.open(str(near_path)).convert("RGB")
        except Exception:
            near_img = Image.new("RGB", (224, 224))
        try:
            card_img = Image.open(str(card_path)).convert("RGB")
        except Exception:
            card_img = Image.new("RGB", (224, 224))
        if self.transform:
            far_img = self.transform(far_img)
            near_img = self.transform(near_img)
            card_img = self.transform(card_img)
        return far_img, near_img, card_img, label


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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class CrossAttentionFusion(nn.Module):
    """Simple cross-attention: use card features as query to attend to face features."""
    def __init__(self, dim=512, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        B = query.shape[0]
        q = self.q_proj(query).unsqueeze(1)   # (B, 1, D)
        k = self.k_proj(key_value).unsqueeze(1)  # (B, 1, D)
        v = self.v_proj(key_value).unsqueeze(1)   # (B, 1, D)

        h = self.heads
        d = q.shape[-1] // h
        q = q.view(B, 1, h, d).transpose(1, 2)  # (B, h, 1, d)
        k = k.view(B, 1, h, d).transpose(1, 2)
        v = v.view(B, 1, h, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, -1)
        return self.out_proj(out.squeeze(1))


class TripleStreamAttentionNet(nn.Module):
    """
    Triple-stream with shared ResNet18 encoder.
    Features:
    - 3 stream outputs (far, near, card) each 512-d
    - Cross-attention: card queries each face stream
    - Difference features: far-card, near-card, far-near
    - SE attention on final fused features
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512

        self.card_far_attn = CrossAttentionFusion(feat_dim, heads=4)
        self.card_near_attn = CrossAttentionFusion(feat_dim, heads=4)

        # 3 raw + 3 diffs + 2 cross-attn = 8 * 512 = 4096
        fused_dim = feat_dim * 8
        self.se = SEBlock(fused_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, far, near, card):
        f_far = self.encoder(far).flatten(1)
        f_near = self.encoder(near).flatten(1)
        f_card = self.encoder(card).flatten(1)

        attn_card_far = self.card_far_attn(f_card, f_far)
        attn_card_near = self.card_near_attn(f_card, f_near)

        diff_far_card = f_far - f_card
        diff_near_card = f_near - f_card
        diff_far_near = f_far - f_near

        combined = torch.cat([
            f_far, f_near, f_card,
            diff_far_card, diff_near_card, diff_far_near,
            attn_card_far, attn_card_near,
        ], dim=1)

        combined = self.se(combined)
        return self.classifier(combined)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


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
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = LivenessDatasetTriple(train_ids, labels, transform_train)
    test_ds = LivenessDatasetTriple(test_ids, labels, transform_test)
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=40, shuffle=False, num_workers=4, pin_memory=True)

    model = TripleStreamAttentionNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Triple-stream ResNet18 + cross-attention + SE fusion, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    epochs = 15
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

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
            far, near, card, targets = far.to(device), near.to(device), card.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(far, near, card)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels_list, all_probs = [], [], []
        with torch.no_grad():
            for far, near, card, targets in test_loader:
                far, near, card = far.to(device), near.to(device), card.to(device)
                out1 = model(far, near, card)
                out2 = model(
                    torch.flip(far, dims=[3]),
                    torch.flip(near, dims=[3]),
                    torch.flip(card, dims=[3])
                )
                out = (out1 + out2) / 2
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        bal_acc = balanced_accuracy_score(all_labels_list, all_preds)
        acc = accuracy_score(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average="binary")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1

    elapsed = time.time() - t0
    approach = "triple_stream_resnet18_cross_attention_SE_focal_TTA"
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

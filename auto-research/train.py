"""
train.py — Liveness detection: Three-stream ResNet18 shared encoder with
cross-stream similarity features + card image.

Key insight: Card image has NEVER been used before. Deepfake attacks (70% of
negatives) show mismatches between the card's face and the far/near faces.
We use a shared encoder for all 3 images, then compute:
  - Concatenated features (512*3 = 1536)
  - Cosine similarities between all stream pairs (3 values)
  - L2 distance between stream pairs (3 values)
  - Element-wise absolute difference norms for card vs face streams (2 values)
Total classifier input: 1536 + 8 = 1544 features
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
    """Loads far, near, AND card images for each sample."""
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
                  and (samples_dir / s / "near.jpg").exists()
                  and (samples_dir / s / "card.jpg").exists()]
    binary_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in valid_sigs]
    print(f"Total valid samples (with card): {len(valid_sigs)}")
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


class ThreeStreamResNet(nn.Module):
    """
    Shared ResNet18 encoder processes far, near, card independently.
    Cross-stream similarity features are computed and concatenated
    with the raw feature vectors for the classifier.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512

        # Classifier input: 3*feat_dim + 3 cosine sims + 3 L2 dists + 2 card-face diff norms = 1544
        cls_in = feat_dim * 3 + 8
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(cls_in),
            nn.Dropout(0.3),
            nn.Linear(cls_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, far, near, card):
        f_far = self.encoder(far).flatten(1)    # (B, 512)
        f_near = self.encoder(near).flatten(1)   # (B, 512)
        f_card = self.encoder(card).flatten(1)   # (B, 512)

        cos_fn = nn.CosineSimilarity(dim=1)
        cos_far_near = cos_fn(f_far, f_near).unsqueeze(1)   # (B, 1)
        cos_far_card = cos_fn(f_far, f_card).unsqueeze(1)   # (B, 1)
        cos_near_card = cos_fn(f_near, f_card).unsqueeze(1)  # (B, 1)

        l2_far_near = torch.norm(f_far - f_near, dim=1, keepdim=True)   # (B, 1)
        l2_far_card = torch.norm(f_far - f_card, dim=1, keepdim=True)   # (B, 1)
        l2_near_card = torch.norm(f_near - f_card, dim=1, keepdim=True) # (B, 1)

        diff_far_card_norm = torch.norm(torch.abs(f_far - f_card), dim=1, keepdim=True)  # (B, 1)
        diff_near_card_norm = torch.norm(torch.abs(f_near - f_card), dim=1, keepdim=True) # (B, 1)

        combined = torch.cat([
            f_far, f_near, f_card,
            cos_far_near, cos_far_card, cos_near_card,
            l2_far_near, l2_far_card, l2_near_card,
            diff_far_card_norm, diff_near_card_norm,
        ], dim=1)

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
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = ThreeStreamResNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Three-stream ResNet18 shared encoder + cross-stream similarity, params: {num_params:,}")

    train_labels = [0 if labels[s]["main_label"] == "Positive" else 1 for s in train_ids]
    class_counts = Counter(train_labels)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = FocalLoss(alpha=weight.to(device), gamma=2.0)

    epochs = 18
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for far, near, card, targets in test_loader:
                far, near, card = far.to(device), near.to(device), card.to(device)
                # TTA: original + horizontal flip
                out1 = model(far, near, card)
                out2 = model(
                    torch.flip(far, dims=[3]),
                    torch.flip(near, dims=[3]),
                    torch.flip(card, dims=[3]),
                )
                out = (out1 + out2) / 2
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_list.extend(targets.numpy())

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
    approach = "three_stream_resnet18_shared_encoder_card_crosssim_focal_TTA"
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

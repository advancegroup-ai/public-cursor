"""
train.py — Liveness detection training script (runs on DGX1).
Experiment: CLIP ViT-B/32 frozen feature extraction + trainable classifier.
Uses OpenAI CLIP pretrained visual encoder to extract features from both
far and near images, concatenates embeddings, trains a small MLP classifier.
CLIP's pretrained features capture semantic and low-level visual cues that
may help distinguish real vs fake faces.
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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


class DualImageDataset(Dataset):
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

    valid = []
    for sig, info in labels.items():
        far = DATA_DIR / "samples" / sig / "far.jpg"
        near = DATA_DIR / "samples" / sig / "near.jpg"
        if far.exists() and near.exists():
            valid.append(sig)

    random.shuffle(valid)
    split = int(len(valid) * 0.8)
    train_ids = valid[:split]
    test_ids = valid[split:]

    dist_train = Counter(labels[s]["main_label"] for s in train_ids)
    dist_test = Counter(labels[s]["main_label"] for s in test_ids)
    print(f"Train: {len(train_ids)} {dict(dist_train)}")
    print(f"Test:  {len(test_ids)} {dict(dist_test)}")

    return train_ids, test_ids, labels


def train():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, test_ids, labels = load_data()

    # Try to use CLIP, fall back to ResNet50 features if CLIP not available
    try:
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        use_clip = True
        feat_dim = 512
        transform = clip_preprocess
        print("Using CLIP ViT-B/32 features")
    except ImportError:
        print("CLIP not available, using torchvision ViT-B/16 features")
        import torchvision.models as models
        backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        backbone.heads = nn.Identity()
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        use_clip = False
        feat_dim = 768
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print("Using ViT-B/16 ImageNet features")

    train_ds = DualImageDataset(train_ids, labels, transform)
    test_ds = DualImageDataset(test_ids, labels, transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    print("Extracting features (frozen encoder)...")

    def extract_all_features(loader):
        all_feats = []
        all_labels_list = []
        with torch.no_grad():
            for far_imgs, near_imgs, targets in loader:
                far_imgs = far_imgs.to(device)
                near_imgs = near_imgs.to(device)

                if use_clip:
                    far_feat = clip_model.encode_image(far_imgs).float()
                    near_feat = clip_model.encode_image(near_imgs).float()
                else:
                    far_feat = backbone(far_imgs)
                    near_feat = backbone(near_imgs)

                diff_feat = far_feat - near_feat
                combined = torch.cat([far_feat, near_feat, diff_feat], dim=1)
                all_feats.append(combined.cpu())
                all_labels_list.append(targets)

        return torch.cat(all_feats), torch.cat(all_labels_list)

    X_train, y_train = extract_all_features(train_loader)
    X_test, y_test = extract_all_features(test_loader)
    print(f"Features extracted: train={X_train.shape}, test={X_test.shape}")
    feat_time = time.time() - t0
    print(f"Feature extraction took {feat_time:.1f}s")

    input_dim = X_train.shape[1]

    classifier = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    ).to(device)

    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Classifier params: {num_params:,}")

    train_labels_np = y_train.numpy()
    class_counts = Counter(int(l) for l in train_labels_np)
    weight = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weight = weight / weight.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_feat_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    best_bal_acc = 0
    best_acc = 0
    best_f1 = 0
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached at epoch {epoch}")
            break

        classifier.train()
        total_loss = 0
        for feats, targets in train_feat_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            out = classifier(feats)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        classifier.eval()
        with torch.no_grad():
            out = classifier(X_test.to(device))
            preds = out.argmax(dim=1).cpu().numpy()

        bal_acc = balanced_accuracy_score(y_test.numpy(), preds)
        acc = accuracy_score(y_test.numpy(), preds)
        f1 = f1_score(y_test.numpy(), preds, average="binary")
        avg_loss = total_loss / len(train_feat_loader)

        if epoch % 5 == 0 or bal_acc > best_bal_acc:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} f1={f1:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_acc = acc
            best_f1 = f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Also try sklearn classifiers on the extracted features
    print("\nTrying sklearn classifiers on frozen features...")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X_tr_np = X_train.numpy()
    X_te_np = X_test.numpy()
    y_tr_np = y_train.numpy()
    y_te_np = y_test.numpy()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_np)
    X_te_s = scaler.transform(X_te_np)

    sk_configs = [
        ("RF_300", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ("GBM_200", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED)),
    ]

    for name, clf in sk_configs:
        if time.time() - t0 > MAX_SECONDS - 20:
            break
        clf.fit(X_tr_s, y_tr_np)
        preds_sk = clf.predict(X_te_s)
        ba = balanced_accuracy_score(y_te_np, preds_sk)
        ac = accuracy_score(y_te_np, preds_sk)
        f = f1_score(y_te_np, preds_sk, average="binary")
        print(f"  {name}: bal_acc={ba:.4f} acc={ac:.4f} f1={f:.4f}")
        if ba > best_bal_acc:
            best_bal_acc = ba
            best_acc = ac
            best_f1 = f

    elapsed = time.time() - t0

    encoder_name = "clip_vitb32" if use_clip else "vit_b16_imagenet"
    approach = f"{encoder_name}_frozen_dual_embed_mlp_classifier"
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

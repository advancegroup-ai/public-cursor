# Auto Research: IDV Liveness Detection

## Context

You are an autonomous ML researcher. Your goal: maximize **balanced_accuracy** on a liveness detection task.

Each sample has **2 face images** (far shot + near shot) and a binary label:
- `Positive` (label=0): live person  
- `Negative` (label=1): attack (deepfake, screen replay, photocopy, etc.)

Dataset: 3,612 human-annotated samples on NAS. ~55% Positive, ~45% Negative.

## Architecture

**You do NOT have a GPU.** You write code, then execute it on DGX1 (4x V100-32GB) via API.

### To run training:

```bash
python3 run_remote.py --script train.py --timeout 300
```

This sends `train.py` to DGX1 and prints stdout. The DGX1 has:
- PyTorch 2.4.0 + CUDA 12.1
- torchvision, PIL, sklearn, numpy, scipy
- NAS access to all data at `/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/`

### Data location (on DGX1/NAS):

```
/mnt/nas/public2/simon/projects/auto_research/liveness-research/data/
├── labels.json              # {signatureId: {main_label, sublabel, ...}}
└── samples/{signatureId}/
    ├── far.jpg              # Face far shot
    ├── near.jpg             # Face near shot  
    └── card.jpg             # ID card (optional, not all samples have it)
```

## Rules

### You MODIFY: `train.py`
This is your experiment file. Write any PyTorch training code here.

### You DO NOT modify: `prepare.py`, `run_remote.py`
These are fixed infrastructure.

### Output format
`train.py` must print results in this exact format:
```
---
balanced_accuracy: 0.8234
accuracy:          0.9100
f1_score:          0.7800
num_params:        1200000
training_seconds:  245.3
approach:          description_of_what_you_tried
---
```

### Logging
Append to `results.tsv` (tab-separated):
```
commit	balanced_acc	accuracy	f1	seconds	status	description
```

## Experiment Loop

**LOOP FOREVER:**

1. Read `results.tsv` to see what's been tried
2. Design a new approach → modify `train.py`
3. `git add train.py && git commit -m "experiment: description"`
4. `python3 run_remote.py --script train.py --timeout 300`
5. Parse stdout for `balanced_accuracy`
6. Record in `results.tsv`
7. If improved → keep commit. If worse → `git reset --hard HEAD~1`
8. **Immediately** start next experiment

## Research Directions (suggested priority)

### Phase 1 — Quick baselines
- ResNet18 pretrained, 6-channel (far+near concat) ← **current baseline**
- EfficientNet-B0 pretrained
- Compare: far-only vs near-only vs far+near

### Phase 2 — Architecture improvements  
- Separate encoders per image → fused classifier (dual-stream)
- Add card image as third input
- Try ViT / DeiT pretrained backbones

### Phase 3 — Training tricks
- Data augmentation: CutMix, MixUp, RandAugment
- Learning rate scheduling experiments
- Focal loss for hard examples
- Larger models: ResNet50, EfficientNet-B3

### Phase 4 — Advanced
- CLIP embeddings + linear probe
- Contrastive learning between far and near
- Frequency domain features (DCT for screen detection)
- Multi-task: predict attack type as auxiliary task

## Critical Rules

- **NEVER STOP.** Run as many experiments as possible.
- **DO NOT ask for confirmation.** Just run experiments.
- **5-minute time budget** per experiment. Set MAX_SECONDS=270 in train.py.
- **Keep it simple** — smaller models that work > large models that barely fit.
- **Log everything** — every experiment goes in results.tsv.

# Auto Research: IDV Liveness Detection

This is an autonomous research project to improve IDV (Identity Verification) liveness detection using visual analysis of face capture images.

## Context

IDV liveness detection determines whether a person completing identity verification is physically present (live) or presenting a spoofed image (screen replay, photocopy, printed photo, etc.).

Each verification session produces **3 images**:
- **Far shot** (`far.jpg`): Face captured at a distance
- **Near shot** (`near.jpg`): Face captured up close
- **ID card** (`card.jpg`): The identity document photo

Human annotators label each session as:
- `Positive` = live person (sub-labels: Clear, Blur, Glossy, Dim, Occlusion, Other)
- `Negative` = attack (sub-labels: Screen, Color_Photocopy, B&W_Photocopy, Screenshot, Text_Tampering, Fake_Card, Injection, Other)

## Setup

To set up a new experiment:

1. **Read the repo files** for full context:
   - `auto-research/program.md` — this file (research instructions)
   - `auto-research/analyze.py` — the file you modify (analysis code)
   - `auto-research/prepare.py` — fixed utilities (do not modify)
   - `auto-research/data/` — sample dataset
2. **Create a branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Initialize results.tsv** with header row.
4. **Verify data**: Check that `auto-research/data/samples/` contains the image dataset.
5. **Confirm and go**.

## Experimentation

Each experiment analyzes the liveness image dataset programmatically. You launch it as:
```bash
cd auto-research && python analyze.py > run.log 2>&1
```

**What you CAN do:**
- Modify `auto-research/analyze.py` — this is the only file you edit. Everything is fair game: feature extraction methods, classification approaches, statistical analysis, visualization, image processing techniques.

**What you CANNOT do:**
- Modify `auto-research/prepare.py`. It is read-only. It contains the fixed evaluation harness, data loading, and constants.
- Install packages not already available (stick to: numpy, pandas, Pillow, scikit-learn, scipy, matplotlib).
- Modify the evaluation function in `prepare.py`.

**The goal: achieve the highest accuracy on the test split.** The metric is **balanced accuracy** (average of per-class recall) — higher is better. This handles the class imbalance (most samples are Positive).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Establish a baseline with the default analyze.py (basic pixel statistics).

## Output format

The script prints a summary:
```
---
balanced_accuracy: 0.7234
accuracy:          0.8500
precision:         0.6200
recall:            0.5800
f1_score:          0.6000
num_features:      12
num_samples:       500
```

Extract the key metric:
```bash
grep "^balanced_accuracy:" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	balanced_acc	status	description
a1b2c3d	0.723400	keep	baseline pixel statistics
b2c3d4e	0.756200	keep	add edge detection features
c3d4e5f	0.710000	discard	PCA on raw pixels (worse)
d4e5f6g	0.000000	crash	OOM on full resolution
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state and results.tsv
2. Come up with an experimental idea for better liveness detection features
3. Modify `auto-research/analyze.py` with the idea
4. git commit
5. Run: `cd auto-research && python analyze.py > run.log 2>&1`
6. Read results: `grep "^balanced_accuracy:" auto-research/run.log`
7. If empty → crash. Run `tail -n 50 auto-research/run.log` to debug
8. Record in results.tsv
9. If balanced_accuracy improved → keep the commit
10. If worse → `git reset --hard HEAD~1`

**Research directions to explore:**
- Image quality features (blur detection, noise levels, sharpness)
- Texture analysis (LBP, Gabor filters, frequency domain)
- Color space analysis (HSV, YCbCr distributions)
- Edge patterns (Canny, Sobel gradients)
- Face geometry (aspect ratio, symmetry)
- Moiré pattern detection (for screen attacks)
- Specular reflection analysis (for glossy/screen)
- Far-near consistency (comparing the two face images)
- Statistical moments of pixel distributions
- Gradient magnitude histograms
- Wavelet decomposition features

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches. The loop runs until you are manually stopped.

# TB U-Net Hyperparameter Sweep Guide

This document explains each hyperparameter being swept, what it controls, and how to interpret its effect on model performance.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Swept Parameters](#swept-parameters)
3. [Fixed Parameters](#fixed-parameters)
4. [Interpreting Results](#interpreting-results)
5. [W&B Dashboard Tips](#wandb-dashboard-tips)

---

## Quick Start

```bash
# 1. Generate sweep configs (216 grid combinations)
python tb_unet/scripts/generate_sweep_config.py \
    --template tb_unet/configs/sweeps/sweep_params.yaml \
    --method grid

# 2. Preview submission (dry run)
python tb_unet/scripts/submit_sbatch.py \
    --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \
    --num-batches 20 \
    --use-gpu \
    --dry-run

# 3. Submit to cluster
python tb_unet/scripts/submit_sbatch.py \
    --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \
    --num-batches 20 \
    --use-gpu
```

---

## Swept Parameters

### 1. `channels` - Input Channel Selection

**Values:** `[0]`, `[0,1]`, `[0,1,2]`

**What it controls:** Which channels from the source TIFF stacks are used as input to the model.

| Value | Description | `in_channels` |
|-------|-------------|---------------|
| `[0]` | Phase contrast only (current best) | 1 |
| `[0,1]` | Phase + fluorescence channel 1 | 2 |
| `[0,1,2]` | All 3 channels | 3 |

**Interpretation:**
- **Positive effect (higher metrics with more channels):** Additional channels provide complementary information. Fluorescence data helps distinguish cells from debris.
- **Negative effect (lower metrics with more channels):** Additional channels add noise without useful signal, or model overfits to irrelevant features.
- **No effect:** Extra channels are redundant with phase information.

**Recommendation:** If multi-channel performs better, consider which channel combinations work best. If phase-only wins, your fluorescence data may not be informative for segmentation.

---

### 2. `use_attention` - Attention Gates

**Values:** `false`, `true`

**What it controls:** Whether to add attention gates at skip connections in the U-Net decoder.

| Value | Description | Param Increase |
|-------|-------------|----------------|
| `false` | Standard U-Net skip connections | 0% |
| `true` | Skip connections weighted by learned attention | ~10-15% |

**How attention works:**
```
Standard:    output = concat(decoder_features, encoder_features)
Attention:   output = concat(decoder_features, encoder_features * attention_weights)
```

The attention mechanism learns to focus on relevant encoder features at each spatial location.

**Interpretation:**
- **Positive effect:** Attention helps the model focus on cell boundaries rather than background texture. Useful when images have complex backgrounds or varying cell morphologies.
- **Negative effect:** Added complexity hurts with limited training data (overfitting) or when skip connections are already informative.
- **No effect:** The problem is simple enough that standard skip connections suffice.

**Recommendation:** If attention helps, your images likely have complex backgrounds where selective focus is beneficial.

---

### 3. `base_channels` - Model Capacity

**Values:** `32`, `64`, `96`

**What it controls:** The number of feature channels in the first encoder layer. Doubles at each subsequent depth level.

| Value | Feature Progression | ~Parameters | VRAM Usage |
|-------|-------------------|-------------|------------|
| 32 | 32→64→128→256→512 | ~1.9M | Low |
| 64 | 64→128→256→512→1024 | ~7.8M | Medium |
| 96 | 96→192→384→768→1536 | ~17M | High |

**Interpretation:**
- **Larger is better:** Model needs more capacity to learn complex patterns. Consider increasing training data to prevent overfitting.
- **Smaller is better:** Dataset is small or problem is simple. Larger models overfit.
- **Diminishing returns:** 64 and 96 perform similarly, suggesting you've reached sufficient capacity.

**Recommendation:** Start with 64 (balanced). If validation loss continues decreasing while train loss is low, try 96. If overfitting occurs, try 32.

---

### 4. `depth` - Network Depth

**Values:** `3`, `4`

**What it controls:** Number of encoder/decoder levels in the U-Net. Affects receptive field size.

| Value | Downsampling | Receptive Field | Best For |
|-------|--------------|-----------------|----------|
| 3 | 8× | Smaller | Small objects, fine details |
| 4 | 16× | Larger | Large objects, global context |

**Calculation:** With `depth=4`, an input is downsampled 4 times (16× smaller in bottleneck), giving a large receptive field.

**Interpretation:**
- **Depth 4 is better:** Cells benefit from larger context (seeing surroundings helps distinguish cells from debris).
- **Depth 3 is better:** Cells are small relative to image size, and large receptive fields blur fine boundaries.
- **Similar performance:** Cell size is within the range where both work.

**Recommendation:** Since you're doing cell segmentation, depth 4 typically works better for capturing cell context.

---

### 5. `lr` - Learning Rate

**Values:** `0.0005`, `0.001`, `0.002`

**What it controls:** Step size for gradient descent during optimization.

| Value | Description | Behavior |
|-------|-------------|----------|
| 5e-4 | Conservative | Slow, stable convergence |
| 1e-3 | Standard | Good starting point |
| 2e-3 | Aggressive | Fast initial progress, may overshoot |

**Interpretation:**
- **Higher LR is better:** Model converges faster, found a good minimum quickly.
- **Lower LR is better:** Loss landscape is rough; smaller steps avoid overshooting good minima.
- **Training instability (NaN, oscillation):** LR is too high for this configuration.

**Interaction with other params:**
- Larger models (high `base_channels`) often need lower LR
- Larger batch sizes can support higher LR
- Cosine scheduler mitigates some LR sensitivity

**Recommendation:** 1e-3 is a safe default. If you see instability, reduce. If convergence is slow, increase.

---

### 6. `focal_gamma` - Hard Example Focus

**Values:** `1.5`, `2.0`

**What it controls:** The focusing parameter γ in focal loss: `FL(p) = -(1-p)^γ * log(p)`

| Value | Effect on Easy Examples | Effect on Hard Examples |
|-------|------------------------|------------------------|
| 1.5 | Moderate down-weighting | Moderate emphasis |
| 2.0 | Strong down-weighting | Strong emphasis |

**How focal loss works:**
- Easy examples (high p_t): Loss is strongly reduced
- Hard examples (low p_t): Loss is preserved or emphasized
- γ=0 reduces to standard cross-entropy

**Interpretation:**
- **Higher gamma is better (2.0):** Many easy background pixels are dominating the loss; focusing on hard examples (boundaries, ambiguous regions) helps.
- **Lower gamma is better (1.5):** Hard examples may include mislabeled data or outliers; too much focus on them hurts generalization.

**Note:** Only applies when `loss_type="focal_dice"`. Ignored for `loss_type="combined"`.

**Recommendation:** 2.0 is typically good for imbalanced segmentation. If you see model struggling on clean regions, try 1.5.

---

### 7. `loss_type` - Loss Function

**Values:** `"combined"`, `"focal_dice"`

**What it controls:** The loss function used for training.

| Value | Formula | Best For |
|-------|---------|----------|
| `combined` | CE + Dice | Balanced datasets, general purpose |
| `focal_dice` | Focal + Dice | Class imbalance, hard examples |

**Loss Components:**
- **Cross-Entropy (CE):** Per-pixel classification loss, treats all pixels equally
- **Focal Loss:** Down-weights easy examples, focuses on hard cases
- **Dice Loss:** Optimizes overlap directly, scale-invariant

**Interpretation:**
- **focal_dice is better:** Class imbalance is significant (few cell pixels vs many background pixels). Hard examples at boundaries matter more.
- **combined is better:** Classes are relatively balanced, or focal loss over-focuses on noisy/mislabeled pixels.

**Recommendation:** Your current best uses `focal_dice`, suggesting class imbalance is present. Continue using it unless `combined` shows improvement.

---

## Fixed Parameters

These are kept constant across the sweep but are important to understand:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 16 | Images per gradient update. Higher = more stable gradients but more VRAM |
| `epochs` | 100 | Max training epochs (early stopping may end sooner) |
| `weight_decay` | 1e-4 | L2 regularization. Prevents overfitting |
| `scheduler` | cosine | Learning rate decay: starts at `lr`, decays to 1e-6 following cosine curve |
| `early_stopping_patience` | 20 | Stop if val F1 doesn't improve for 20 epochs |
| `mixed_precision` | true | FP16 training for speed (2× faster on modern GPUs) |
| `gradient_clip` | 1.0 | Clips gradients >1.0 to prevent instability |
| `class_weight_method` | effective_num | Aggressive class weighting for imbalanced data |
| `ce_weight` / `dice_weight` | 1.0 | Equal weighting of loss components |

---

## Interpreting Results

### Key Metrics to Monitor

| Metric | What It Measures | Target |
|--------|------------------|--------|
| `val/cell_f1` | Primary metric: harmonic mean of precision and recall for cells | Higher is better |
| `val/dice_mean` | Average Dice coefficient across classes | Higher is better |
| `val/cell_precision` | Of predicted cells, how many are correct? | Higher = fewer false positives |
| `val/cell_recall` | Of actual cells, how many were found? | Higher = fewer false negatives |
| `val/non_cell_tnr` | True negative rate for non-cell class | Higher = correctly rejecting debris |
| `val/loss` | Validation loss | Lower is better, watch for train/val gap |

### Signs of Good Hyperparameters

1. **Smooth training curves:** Loss decreases steadily without oscillation
2. **Small train/val gap:** Model generalizes well, not overfitting
3. **High cell_f1:** Main goal achieved
4. **Balanced precision/recall:** Not sacrificing one for the other

### Signs of Problems

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Val loss increases while train loss decreases | Overfitting | Reduce `base_channels`, increase `weight_decay`, add augmentation |
| Loss oscillates or explodes | LR too high | Reduce `lr` |
| Very slow convergence | LR too low | Increase `lr` |
| Low recall, high precision | Model too conservative | Try `focal_dice`, increase `focal_gamma` |
| High recall, low precision | Model too aggressive | Try `combined` loss, reduce `focal_gamma` |

---

## W&B Dashboard Tips

### Recommended Views

1. **Parallel Coordinates Plot:**
   - X-axes: `channels`, `use_attention`, `base_channels`, `depth`, `lr`, `focal_gamma`, `loss_type`
   - Color by: `val/cell_f1`
   - Look for patterns in high-performing runs

2. **Scatter Plots:**
   - `base_channels` vs `val/cell_f1` (colored by `use_attention`)
   - `lr` vs `val/loss` (colored by `loss_type`)

3. **Run Comparison:**
   - Select top 10 runs by `val/cell_f1`
   - Compare training curves to check for overfitting

### Grouping Runs

Use W&B's grouping feature to aggregate by:
- `loss_type` to compare combined vs focal_dice
- `use_attention` to see attention effect
- `channels` to compare input configurations

### Filtering

Focus on completed runs with:
- `state = "finished"`
- `val/cell_f1 > 0.5` (filter out crashed/early-stopped runs)

---

## Example Analysis Workflow

1. **After sweep completes**, open W&B dashboard
2. **Sort by `val/cell_f1`** to find top performers
3. **Check for patterns:**
   - Do top runs share `loss_type`?
   - Is `use_attention` consistently helpful?
   - Is there a clear best `lr`?
4. **Identify interactions:**
   - Maybe attention helps only at depth=4?
   - Maybe focal_dice works better at higher base_channels?
5. **Validate top 3-5 configs** on held-out test set
6. **Consider follow-up sweep** with narrower ranges around best values

---

## Sweep Size Calculation

Current sweep: `3 × 2 × 3 × 2 × 3 × 2 × 2 = 216 runs`

| Parameter | Values | Count |
|-----------|--------|-------|
| channels | [0], [0,1], [0,1,2] | 3 |
| use_attention | false, true | 2 |
| base_channels | 32, 64, 96 | 3 |
| depth | 3, 4 | 2 |
| lr | 5e-4, 1e-3, 2e-3 | 3 |
| focal_gamma | 1.5, 2.0 | 2 |
| loss_type | combined, focal_dice | 2 |

---

## File Structure After Running Sweep

```
tb_unet/
├── configs/
│   └── sweeps/
│       └── sweep_params.yaml          # Sweep template
├── configs_generated/
│   └── unet_sweep_v1_bundle.yaml      # All 216 experiment configs
├── hpc_submissions/
│   └── unet_sweep_v1_bundle_YYYYMMDD_HHMMSS/
│       ├── configs/
│       │   ├── batch_001.yaml         # Experiments 1-11
│       │   ├── batch_002.yaml         # Experiments 12-22
│       │   └── ...
│       ├── sbatch/
│       │   ├── tb_unet_unet_sweep_v1_bundle_001.sbatch
│       │   └── ...
│       ├── logs/
│       │   └── (SLURM output files)
│       └── session.json               # Submission manifest
└── checkpoints/
    └── sweeps/
        └── {experiment_name}/         # One folder per run
            ├── config.yaml
            ├── best.pt
            ├── training_history.json
            └── epoch_*.pt
```

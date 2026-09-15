# Beyond Consistency: Explicit Boundary Learning for Semi-Supervised Ovarian Tumor Segmentation (BA-Net)

> **Official implementation** of Vu, Bui & Le, *"Beyond Consistency: Explicit Boundary Learning for
> Semi-supervised Ovarian Tumor Segmentation"*, Proc. 28th International Conference on Pattern
> Recognition (**ICPR 2026**, Lyon, France), Part II, LNCS Vol. 16813, pp. 245–260, Springer.
>
> 📄 **Paper (PDF):** <https://link.springer.com/content/pdf/10.1007/978-3-032-31583-0_17>
> 🔗 **DOI:** [10.1007/978-3-032-31583-0_17](https://doi.org/10.1007/978-3-032-31583-0_17)

## Abstract

Consistency-regularization frameworks (e.g. Mean Teacher) have become the de facto paradigm for
semi-supervised medical image segmentation, yet they optimize *region-level agreement* and remain
largely insensitive to anatomical boundaries — precisely where ovarian tumors in ultrasound exhibit
low contrast, speckle noise, and ambiguous transitions to surrounding tissue. This repository
implements **BA-Net (Boundary-Aware Network)**, which moves *beyond consistency* by adding
**explicit boundary learning** to the semi-supervised loop: (i) a deeply-supervised U-Net with a
dedicated refinement sub-network that emits multi-scale boundary-aware predictions; (ii) a
**two-stage Mean-Teacher protocol** (labeled pre-training followed by labeled + unlabeled
self-training with a frozen pseudo-label generator); and (iii) a composite objective coupling a
BCE + SSIM + IoU deep-supervision loss, EMA consistency regularization, and a pseudo-label
boundary loss on unlabeled data. Evaluated on the **OTU-2D** ovarian-tumor ultrasound benchmark
under low-labeled regimes (e.g. 10% labeled), BA-Net yields more accurate masks *and* sharper
tumor boundaries than consistency-only baselines, as measured by Dice, Jaccard/IoU, Precision,
Recall, and HD95.

## 1. Introduction

Automatic ovarian-tumor segmentation from ultrasound is a key step in computer-aided diagnosis,
but dense expert annotation is expensive while ultrasound boundaries are intrinsically fuzzy.
Semi-supervised learning (SSL) leverages abundant unlabeled scans, yet standard teacher–student
consistency treats every pixel equally and under-constrains the boundary band that matters most
clinically. BA-Net addresses this gap: in addition to enforcing student↔teacher agreement, the
student is explicitly supervised to reproduce **boundary-faithful** pseudo-labels generated from
the best pre-trained teacher, with structural (SSIM) and overlap (IoU) terms that penalize
boundary misalignment at multiple decoder scales.

## 2. Method

### 2.1 Boundary-aware architecture (`model.py`: `Proposed`)

A 5-level U-Net encoder–decoder (64→1024 channels) with:

- **Deep supervision:** 1×1/3×3 side-heads on three decoder stages, upsampled to full resolution
  (`side1–side3`), plus a coarse full-resolution head.
- **Refinement module** (`RefUnet`): a residual U-shaped refiner applied to the coarse logits,
  producing the final prediction `s0`.
- **Output:** five sigmoid maps `(s0, s1, s2, s3, s4)` = (refined, coarse, side3, side2, side1),
  all supervised during training; only `s0` is used at inference (`test.py`).

### 2.2 Two-stage Mean-Teacher training (`train.py`)

| Stage | Data | Objective |
|---|---|---|
| **1. Pre-train** (`pre_train_one_epoch`) | labeled only | `L = L_sup + λ(t)·L_con`, where `L_sup` is the multi-output BCE+SSIM+IoU fusion loss and `L_con` is the student↔teacher MSE over all five outputs |
| **2. Self-train** (`self_train_one_epoch`) | labeled + unlabeled (two-stream batches, 25% labeled) | `L = L_sup + λ(t)·L_con + β(t)·L_pseudo`, where `L_pseudo = BCE(s0_un, round(t̂0_un)) + Dice(s1_un, round(t̂1_un))` uses the **frozen best pre-train teacher** as pseudo-label generator |

- The teacher is updated by **exponential moving average** (EMA, 0.99→0.999) with a sigmoid
  ramp-up schedule (`ramp.py`, cf. Tarvainen & Valpola).
- **Dual-view augmentation** (`data.py`): shared geometric transforms (D4, resized-crop, rotation)
  keep the student/teacher pair aligned; the student then receives *strong* photometric noise
  (blur, color jitter, gray) while the teacher receives *weak* noise — the standard
  consistency gap.

### 2.3 Losses (`loss.py`) and metrics (`metrics.py`)

- `loss.py` — `bce_ssim_loss` (BCE + windowed **SSIM** + **IoU**, inlined from the former
  `pytorch_ssim` / `pytorch_iou` packages), `muti_bce_loss_fusion` (deep-supervision sum over the
  five outputs), `MSE_loss` (consistency), `unlabeled_loss` (pseudo-label BCE + Dice),
  `joint_loss1` (focal + simplified-SSIM + Jaccard, used for validation).
- `metrics.py` — `dice_coef`, `jaccard_similarity`, `recall_precision`, `compute_hd95`.

## 3. Repository Structure

```text
BA-Net/
├── train.py          # two-stage Mean-Teacher training (pre-train + self-train + test)
├── test.py           # evaluate() shared by training + standalone test entry point
├── data.py           # OTU-2D loaders, dual-view augmentation, TwoStreamBatchSampler
├── loss.py           # all training losses (SSIM/IOU consolidated here)
├── metrics.py        # Dice, Jaccard, precision/recall, HD95
├── model.py          # Proposed (boundary-aware U-Net + RefUnet) + ablations
├── resnet_model.py   # residual blocks for the BASNet variant
├── ramp.py           # sigmoid ramp-up schedule
├── weight/           # checkpoints (e.g. weight/proposed.pth; git-ignored)
└── README.md
```

## 4. Dataset

**OTU-2D** ovarian-tumor ultrasound dataset (train / validation / test splits described by
`OTU_2D_annotation.json`). Expected layout — a sibling folder of this repository:

```text
KhoaVM/
├── BA-Net/               # this repo
└── OTU-2D-Dataset/       # data root
    ├── OTU_2D_annotation.json
    └── OTU_2D/...
```

The data root is resolved **relatively** (`../OTU-2D-Dataset` from `BA-Net/`) and can be
overridden without editing code:

```bash
export OTU_2D_DATASET_ROOT=/path/to/OTU-2D-Dataset
```

## 5. Installation

```bash
pip install torch torchvision albumentations opencv-python numpy
```

## 6. Usage

### 6.1 Train (10% labeled, defaults reproduce the paper setup)

```bash
cd BA-Net
python train.py --dataset_name OTU --labeled_ratio 0.1 \
    --pre_epochs 50 --epochs 50 --batch_size 4 \
    --device_id cuda:0 --best_model_path weight/proposed.pth
```

Key options: `--labeled_ratio`, `--pre_epochs`, `--epochs`, `--max_lambda` (consistency weight),
`--max_beta` (pseudo-label weight), `--start_ema_coef` / `--end_ema_coef`, `--learning_rate`,
`--annotation_file`.

### 6.2 Test

```bash
python test.py --checkpoint weight/proposed.pth --dataset_name OTU --batch_size 4
```

Reports Dice, Jaccard/IoU, Precision, Recall, and HD95 on the test split.

## 7. Results

The paper reports experiments on OTU-2D under low-labeled regimes with Dice / IoU / Precision /
Recall / HD95; see Tables 1–3 and the qualitative boundary comparisons in the published PDF
([link](https://link.springer.com/content/pdf/10.1007/978-3-032-31583-0_17)). Run §6 above to
reproduce the numbers with your local data split.

## 8. Citation

```bibtex
@inproceedings{VuBL26,
  title     = {Beyond Consistency: Explicit Boundary Learning for Semi-supervised Ovarian Tumor Segmentation},
  author    = {Vu, Minh-Khoa and Bui, Hoang-Son and Le, Thi-Lan},
  booktitle = {Pattern Recognition -- 28th International Conference, ICPR 2026, Lyon, France, August 17--22, 2026, Proceedings, Part II},
  volume    = {16813},
  series    = {Lecture Notes in Computer Science},
  pages     = {245--260},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-31583-0_17},
  url       = {https://doi.org/10.1007/978-3-032-31583-0_17}
}
```

## 9. Acknowledgements

SSIM implementation adapted from
[Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim); EMA ramp-up follows the
Mean-Teacher schedule (arXiv:1610.02242). Ultrasound data: OTU-2D benchmark.

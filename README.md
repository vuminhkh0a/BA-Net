# Beyond Consistency: Explicit Boundary Learning for Semi-supervised Ovarian Tumor Segmentation (BA-Net)

Minh-Khoa Vu<sup>1,2</sup>, Hoang-Son Bui<sup>1,2</sup>, and Thi-Lan Le<sup>1,2 (✉)</sup>
<sup>1</sup>SigM Laboratory, School of Electrical and Electronic Engineering, Hanoi University of
Science and Technology, Hanoi 100000, Vietnam
<sup>2</sup>School of Electrical and Electronic Engineering, Hanoi University of Science and
Technology, Hanoi 100000, Vietnam
✉ lan.lethi1@hust.edu.vn

> Proc. 28th International Conference on Pattern Recognition (**ICPR 2026**, Lyon, France),
> Part II, LNCS Vol. 16813, pp. 245–260, Springer.
>
> 📄 [Paper (PDF)](https://link.springer.com/content/pdf/10.1007/978-3-032-31583-0_17) · 🔗 [DOI](https://doi.org/10.1007/978-3-032-31583-0_17) · 💻 [Source code](https://github.com/vuminhkh0a/BA-Net)
>
> **Keywords:** Boundary refinement · Ovarian tumor · Segmentation · Semi-supervised learning · Pseudo-label

## Abstract

Accurate segmentation of ovarian tumors in ultrasound images is critical for early diagnosis and
risk stratification but remains challenging due to boundary ambiguity, speckle noise, and the high
cost of pixel-level annotation. To address these limitations, we propose a semi-supervised framework
that effectively leverages unlabeled data through a Boundary-Aware Mean Teacher paradigm. Our method
integrates a Boundary Refinement Module to explicitly recover fine-grained structural details often
lost in standard semi-supervised approaches. Specifically, we introduce a coarse-to-fine
pseudo-labeling strategy, where a hybrid Dice-BCE loss is dynamically assigned to coarse predictions
for global structure learning and refined predictions for pixel-level boundary alignment.
Furthermore, we incorporate multi-scale deep supervision with hierarchical consistency to maximize
feature representation across decoder layers. Extensive experiments on the benchmark OTU_2D, OvaTUS
and USOVA3D dataset demonstrate that our method significantly improves boundary delineation and
outperforms state-of-the-art semi-supervised methods, particularly in low-data regimes.

## 1. Motivation

Ovarian cancer is a leading cause of gynecological mortality, and early diagnosis substantially
improves outcomes. Ultrasound is the primary first-line modality for adnexal assessment, but
automated segmentation is confounded by speckle noise, acoustic shadowing, and attenuation, plus
large morphological variability across tumors. Predicted contours must align tightly with true
tumor boundaries because morphology and size feed directly into risk-stratification systems such
as O-RADS. Dense annotation, however, is costly and prone to inter-observer variability.

Consistency regularization — in particular the Mean Teacher (MT) framework, which forms the
teacher as an exponential moving average (EMA) of the student — is the dominant semi-supervised
strategy. Hierarchical variants (HCR-MT) add multi-scale consistency but leave two problems open
for ovarian ultrasound, which this work targets:

1. **Unlabeled data are under-utilized.** Hierarchical consistency uses unlabeled samples only for
   perturbation invariance, with no mechanism to learn explicitly from high-confidence predictions
   via pseudo-supervision — limiting performance when labeled data are extremely scarce.
2. **Standard consistency losses blur boundaries.** Tissue surfaces are discontinuous under acoustic
   shadows, so the teacher produces uncertain/blurry edges and MSE-based consistency lets the
   student inherit them. Unlike uncertainty-filtering approaches that merely discard unreliable
   predictions, BA-Net **actively recovers** boundary detail.

**Contributions.** (i) A Boundary Refinement Module (BRM) that reconstructs high-uncertainty edge
regions; (ii) a hybrid strategy combining hierarchical consistency with a dual-stage
(coarse-to-fine) pseudo-labeling mechanism; (iii) state-of-the-art results on three ovarian-tumor
datasets — e.g. with 10% labeled data, **76.77% DSC on OTU_2D**, ≈10 points above the HCR-MT
baseline (66.85% DSC).

## 2. Method

### 2.1 Problem setting and architecture

Let the training set be D = D<sub>l</sub> ∪ D<sub>u</sub>, with M labeled samples
D<sub>l</sub> = {(x<sup>l</sup><sub>i</sub>, y<sup>l</sup><sub>i</sub>)}<sup>M</sup><sub>i=1</sub> and N unlabeled
samples D<sub>u</sub> = {x<sup>u</sup><sub>i</sub>}<sup>M+N</sup><sub>i=M+1</sub>, M ≪ N. Student (S) and
Teacher (T) share the same architecture: a UNet-based **Prediction Module (PM)**, f(·), followed
by the **Boundary Refinement Module (BRM)**, f<sub>BRM</sub>(·).

- PM decoder side-outputs: ŷ<sub>j</sub> = f<sub>j</sub>(x), j ∈ {1,…,4}, each auxiliary block a
  3×3 convolution + up-sampling + sigmoid.
- Final output: ŷ<sub>BRM</sub> from the BRM.
- In code (`models/banet.py: Proposed`) this is the 5-tuple `(s0, s1, s2, s3, s4)` = (refined BRM output,
  coarse PM output, three upsampled side-outputs); only `s0` is used at inference (`test.py`).

### 2.2 Two-phase training

Built on HCR-MT (deep supervision + consistency at every side-output), with coarse-to-fine
pseudo-labeling added so unlabeled semantics are learned explicitly, not just regularized.

**Phase 1 — Pre-training** (`train.py: pre_train_one_epoch`). Labeled samples pass through
Student and Teacher under distinct perturbations η, η′. The Student's multi-scale predictions are
supervised by L<sub>lab</sub>; the Teacher's predictions serve as consistency targets via
L<sub>cons</sub>. The Teacher follows EMA:

θ′<sub>t</sub> = α·θ′<sub>t−1</sub> + (1 − α)·θ<sub>t</sub>,&nbsp;&nbsp;(1)

with α ramped 0.99 → 0.999 (`utils/ramp.py` sigmoid ramp-up). Objective: L<sub>pre</sub> = L<sub>lab</sub> + λ·L<sub>cons</sub>.

**Phase 2 — Self-training** (`train.py: self_train_one_epoch`). The frozen best pre-training
Teacher (kept as `pseudo_label_generator` in code) converts Teacher predictions on D<sub>u</sub>
into **coarse** pseudo-labels y<sup>u</sup><sub>c</sub> (from the PM) and **refined** pseudo-labels
y<sup>u</sup><sub>r</sub> (from the BRM). Batches are drawn with `TwoStreamBatchSampler` (25%
labeled). The Student optimizes L<sub>lab</sub> + L<sub>unlab</sub> + L<sub>cons</sub> while Teacher
EMA continues. Objective: L<sub>self</sub> = L<sub>lab</sub> + λ·L<sub>cons</sub> + β·L<sub>unlab</sub>.&nbsp;&nbsp;(5)

**Dual-view augmentation** (`data/loader.py`): shared geometric transforms (D4, resized-crop, rotation)
keep the Student/Teacher pair aligned; then *strong* photometric noise for the Student (blur,
color jitter, gray) versus *weak* noise for the Teacher.

### 2.3 Boundary Refinement Module

Adapted from the Residual Refinement Module of BASNet: a lightweight residual encoder–decoder in
which **each stage holds a single 3×3 convolution with fixed width 64** (no channel doubling),
2×2 non-overlapping MaxPool down-sampling, and bilinear up-sampling. The design deliberately
avoids high-level semantic re-learning so the module concentrates on spatial-detail recovery from
the coarse map. In code: `models/banet.py: RefUnet`.

### 2.4 Loss functions

**Supervised + consistency** (`utils/losses.py`). Labeled loss is the BASNet hybrid (pixel-level BCE +
patch-level SSIM + map-level IoU) summed over all five outputs (j = 1…4 PM side-outputs, j = 5 BRM):

L<sub>lab</sub> = Σ<sup>5</sup><sub>j=1</sub> [L<sub>BCE</sub> + L<sub>IoU</sub> + L<sub>SSIM</sub>].&nbsp;&nbsp;(2)

Hierarchical consistency is the squared error between Student and Teacher at every level:

L<sub>cons</sub> = Σ<sup>5</sup><sub>j=1</sub> ‖ŷ<sup>S</sup><sub>j</sub> − ŷ<sup>T</sup><sub>j</sub>‖².&nbsp;&nbsp;(3)

**Unlabeled coarse-to-fine loss** (adapted from Combo Loss: instead of mixing Dice+BCE on one
map, each term supervises its matched output):

L<sub>unlab</sub> = L<sub>DSC</sub>(y<sup>u</sup><sub>c</sub>, ŷ<sup>u,S</sup><sub>4</sub>) + L<sub>BCE</sub>(y<sup>u</sup><sub>r</sub>, ŷ<sup>u,S</sup><sub>BRM</sub>).&nbsp;&nbsp;(4)

Rationale: coarse pseudo-labels carry global structure but boundary noise → Dice enforces
region overlap without punishing pixel-wise edge mismatch; refined pseudo-labels are confident
near boundaries → BCE enforces exact pixel alignment. In code this is `unlabeled_loss`
(`BCE` on the refined output + `dice_loss` on the coarse output); validation uses `joint_loss1`
(focal + simplified-SSIM + Jaccard).

### 2.5 Paper-to-code mapping

| Paper symbol | Code |
|---|---|
| PM f(·), side-outputs ŷ<sub>j</sub>, j∈{1…4} | `models/banet.py: Proposed` decoder + `side_conv1–3`, returned as `s2, s3, s4` (+coarse `s1`) |
| BRM f<sub>BRM</sub>(·), ŷ<sub>BRM</sub> | `models/banet.py: RefUnet`, returned as `s0` |
| L<sub>lab</sub> (Eq. 2) | `utils/losses.py: muti_bce_loss_fusion` (BCE+SSIM+IoU × 5 outputs) |
| L<sub>cons</sub> (Eq. 3) | `utils/losses.py: MSE_loss` summed over 5 outputs |
| L<sub>unlab</sub> (Eq. 4) | `utils/losses.py: unlabeled_loss` |
| EMA (Eq. 1), λ/β ramp-up | `train.py: update_ema_variables`, `utils/ramp.py: sigmoid_rampup` |
| Phase 1 / Phase 2 | `train.py: pre_train_one_epoch` / `self_train_one_epoch` |
| Frozen pre-train Teacher as PL source | `train.py: pseudo_label_generator` |
| Metrics (DSC, IoU, HD95) | `utils/metrics.py`, aggregated by `test.py: evaluate` |

## 3. Repository Structure

```text
BA-Net/
├── train.py              # two-stage Mean-Teacher training (pre-train + self-train + test)
├── test.py               # evaluate() shared by training + standalone test entry point
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── loader.py         # OTU_2D loaders, dual-view augmentation, TwoStreamBatchSampler
├── models/
│   ├── __init__.py
│   ├── banet.py          # Proposed (PM + BRM/RefUnet) + MT/HCRMT/Unet/BASNet variants
│   └── resnet.py         # residual blocks for the BASNet variant
├── utils/
│   ├── __init__.py
│   ├── losses.py         # all training losses (SSIM/IOU consolidated here)
│   ├── metrics.py        # Dice, Jaccard, precision/recall, HD95
│   └── ramp.py           # sigmoid ramp-up schedule
├── checkpoints/          # trained weights (e.g. checkpoints/proposed.pth; git-ignored, LFS)
└── README.md
```

## 4. Datasets

The paper evaluates on three ovarian-tumor ultrasound benchmarks: **OTU_2D**, **OvaTUS**, and
**USOVA3D**. This repository currently wires the **OTU_2D** split via `OTU_2D_annotation.json`.
Expected layout — a sibling folder of this repository:

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
pip install -r requirements.txt
```

## 6. Usage

### 6.1 Train (10% labeled, defaults reproduce the paper setup)

```bash
cd BA-Net
python train.py --dataset_name OTU --labeled_ratio 0.1 \
    --pre_epochs 50 --epochs 50 --batch_size 4 \
    --device_id cuda:0 --best_model_path checkpoints/proposed.pth
```

Key options: `--labeled_ratio`, `--pre_epochs`, `--epochs`, `--max_lambda` (λ, consistency),
`--max_beta` (β, pseudo-label), `--start_ema_coef` / `--end_ema_coef` (α), `--learning_rate`,
`--annotation_file`.

### 6.2 Test

```bash
python test.py --checkpoint checkpoints/proposed.pth --dataset_name OTU --batch_size 4
```

Reports Dice, Jaccard/IoU, Precision, Recall, and HD95 on the test split.

## 7. Citation

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

## 8. Acknowledgements

Funded by the Ministry of Science and Technology (MOST) under grant KC4.0-45/19-25. The BRM
follows the Residual Refinement Module of BASNet; the SSIM implementation is adapted from
[Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim); the EMA ramp-up follows
the Mean-Teacher schedule.

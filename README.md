# BA-Net
## Abstract
Accurate segmentation of ovarian tumors in ultrasound im-
ages is critical for early diagnosis and risk stratification but remains chal-
lenging due to boundary ambiguity, speckle noise, and the high cost of
pixel-level annotation. To address these limitations, we propose a semi-
supervised framework that effectively leverages unlabeled data through
a Boundary-Aware Mean Teacher paradigm. Our method integrates a
Boundary Refinement Module to explicitly recover fine-grained struc-
tural details often lost in standard semi-supervised approaches. Specif-
ically, we introduce a coarse-to-fine pseudo-labeling strategy, where a
hybrid Dice-BCE loss is dynamically assigned to coarse predictions for
global structure learning and refined predictions for pixel-level bound-
ary alignment. Furthermore, we incorporate multi-scale deep supervision
with hierarchical consistency to maximize feature representation across
decoder layers. Extensive experiments on the benchmark OTU_2D, Ova-
TUS and USOVA3D dataset demonstrate that our method significantly
improves boundary delineation and outperforms state-of-the-art semi-
supervised methods, particularly in low-data regimes. Source code will
be made publicly available upon acceptance.

## Keywords
- Ovarian Tumor
- Medical Image Segmentation
- Semi-Supervised Learning
- Boundary Refinement
- Pseudo-labeling

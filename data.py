"""Data loading for BA-Net (OTU-2D ovarian tumor ultrasound).

Dual-view Mean-Teacher pipeline:
  - shared geometric augmentation for the student/teacher pair + mask,
  - strong photometric augmentation for the student, weak for the teacher.
"""
import itertools
import json
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# ---------------------------------------------------------------------------
# Paths (relative to this repository; override with env var if needed)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (REPO_ROOT.parent / "OTU-2D-Dataset").resolve()
DATA_ROOT = Path(os.environ.get("OTU_2D_DATASET_ROOT", str(DEFAULT_DATA_ROOT)))
ANNOTATION_FILE = DATA_ROOT / "OTU_2D_annotation.json"

IMAGE_SIZE = 256

# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
geometry_transform = A.Compose([
    A.D4(p=0.5),
    A.RandomResizedCrop(scale=(0.5, 1.0), size=(IMAGE_SIZE, IMAGE_SIZE), p=0.5),
    A.Rotate(limit=(-15, 15), p=0.5),
], additional_targets={'image2': 'image'})

student_color_transform = A.Compose([
    A.GaussianBlur(p=0.7, blur_limit=10),
    A.ColorJitter(p=0.7, brightness=(0.5, 1.9), contrast=(0.5, 1.9), saturation=(0.5, 1.9), hue=(-0.5, 0.5)),
    A.ToGray(p=0.7),
    A.ToTensorV2(),
])

teacher_color_transform = A.Compose([
    A.GaussianBlur(p=0.1, blur_limit=2),
    A.ColorJitter(p=0.1, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.5, 0.5)),
    A.ToGray(p=0.1),
    A.ToTensorV2(),
])

no_transform = A.Compose([
    A.ToTensorV2(),
])


class Custom_Dataset(Dataset):
    """Returns (student_image, teacher_image, mask, is_labeled)."""

    def __init__(self, images, masks, is_train, labeled_flags=None):
        self.images = images
        self.masks = masks
        self.is_train = is_train
        # Per-sample flag; defaults to True (fully labeled, e.g. val/test).
        if labeled_flags is None:
            labeled_flags = [True] * len(images)
        self.labeled_flags = [bool(v) for v in labeled_flags]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        is_labeled = self.labeled_flags[i]
        image_path = self.images[i]
        image = cv2.resize(
            cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
            (IMAGE_SIZE, IMAGE_SIZE),
        ) / 255.0

        mask_path = self.masks[i]
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(np.where(mask == 0, 0.0, 1.0), -1)

        if self.is_train:
            geo_aug = geometry_transform(image=image.astype(np.float32),
                                         image2=image.astype(np.float32),
                                         mask=mask.astype(np.float32))
            student_image, teacher_image, mask = (
                geo_aug['image'], geo_aug['image2'],
                torch.tensor(geo_aug['mask']).permute(2, 0, 1),
            )

            student_color_aug = student_color_transform(image=student_image)
            teacher_color_aug = teacher_color_transform(image=teacher_image)
            student_image = student_color_aug['image']
            teacher_image = teacher_color_aug['image']
        else:
            t = no_transform(image=image.astype(np.float32), mask=mask.astype(np.float32))
            student_image = t['image']
            teacher_image = t['image']
            mask = t['mask'].permute(2, 0, 1)

        return student_image.float(), teacher_image.float(), mask.float(), is_labeled


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices.

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def _read_splits(name, annotation_file):
    train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    if name == 'OTU':
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        for item in data:
            img = str(DATA_ROOT / item['file_path_img']) if not os.path.isabs(item['file_path_img']) \
                else str(item['file_path_img'])
            ann = str(DATA_ROOT / item['file_path_ann']) if not os.path.isabs(item['file_path_ann']) \
                else str(item['file_path_ann'])
            if item['split'] == 'train':
                train_x.append(img)
                train_y.append(ann)
            elif item['split'] == 'validation':
                valid_x.append(img)
                valid_y.append(ann)
            elif item['split'] == 'test':
                test_x.append(img)
                test_y.append(ann)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def get_datasets(name, labeled_ratio, annotation_file=None):
    annotation_file = Path(annotation_file) if annotation_file else ANNOTATION_FILE
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = _read_splits(name, annotation_file)

    print(f"Dataset: {name}")
    print(f"Training data: {len(train_x)}")
    print(f"Validation data: {len(valid_x)}")
    print(f"Testing data: {len(test_x)}")

    # Random labeled/unlabeled split over the training set.
    n_train = len(train_x)
    n_labeled = int(labeled_ratio * n_train)
    labeled_indices = np.zeros(n_train, dtype=bool)
    labeled_indices[:n_labeled] = True
    np.random.shuffle(labeled_indices)

    labeled_train_dataset = Custom_Dataset(
        [train_x[i] for i, val in enumerate(labeled_indices) if val],
        [train_y[i] for i, val in enumerate(labeled_indices) if val],
        is_train=True,
    )
    train_dataset = Custom_Dataset(train_x, train_y, is_train=True,
                                   labeled_flags=labeled_indices.tolist())
    valid_dataset = Custom_Dataset(valid_x, valid_y, is_train=False)
    test_dataset = Custom_Dataset(test_x, test_y, is_train=False)

    return labeled_train_dataset, train_dataset, valid_dataset, test_dataset


def get_dataloaders(name, batch_size, num_workers, pin_memory, labeled_ratio, annotation_file=None):
    labeled_train_dataset, train_dataset, valid_dataset, test_dataset = get_datasets(
        name, labeled_ratio, annotation_file)

    labeled_flags = np.array(train_dataset.labeled_flags)
    primary_indices = np.where(~labeled_flags)[0].tolist()    # unlabeled
    secondary_indices = np.where(labeled_flags)[0].tolist()   # labeled
    secondary_batch_size = max(1, int(0.25 * batch_size))
    sampler = TwoStreamBatchSampler(primary_indices=primary_indices,
                                    secondary_indices=secondary_indices,
                                    batch_size=batch_size,
                                    secondary_batch_size=secondary_batch_size)

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_dataset, num_workers=num_workers, pin_memory=pin_memory,
                              batch_sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return labeled_train_loader, train_loader, valid_loader, test_loader

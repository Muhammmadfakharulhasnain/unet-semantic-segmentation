from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

# Oxford-IIIT Pet trimap labels:
# raw 1 = pet, raw 2 = background, raw 3 = boundary.
# Model class IDs are contiguous: 0 = pet, 1 = background, 2 = boundary.
RAW_TO_CLASS = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["pet", "background", "boundary"]

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]

class PetSegmentationDataset(Dataset):
    def __init__(self, base_dataset, indices, image_size=128, augment=False):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, mask = self.base_dataset[self.indices[item]]
        image = TF.resize(image, [self.image_size, self.image_size],
                          interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=InterpolationMode.NEAREST)

        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))

        raw_mask = torch.as_tensor(np.array(mask, dtype=np.int64))
        target = torch.zeros_like(raw_mask, dtype=torch.long)
        for raw_value, class_id in RAW_TO_CLASS.items():
            target[raw_mask == raw_value] = class_id

        # Defensive check: unexpected labels should fail loudly, not silently become background.
        valid_raw = torch.zeros_like(raw_mask, dtype=torch.bool)
        for raw_value in RAW_TO_CLASS:
            valid_raw |= (raw_mask == raw_value)
        if not bool(valid_raw.all()):
            bad = torch.unique(raw_mask[~valid_raw]).tolist()
            raise ValueError(f"Unexpected Oxford-IIIT Pet mask labels: {bad}")

        return image, target

def make_splits(dataset_size: int, seed=42,
                train_fraction=0.85, val_fraction=0.15, test_fraction=0.0):
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("Train, validation and test fractions must sum to 1.")
    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(dataset_size * train_fraction)
    n_val = int(dataset_size * val_fraction)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train+n_val].tolist()
    test_idx = indices[n_train+n_val:].tolist()
    return train_idx, val_idx, test_idx

def build_data_bundle(root="data", image_size=128, batch_size=4, num_workers=0,
                      seed=42, augment_train=False,
                      train_fraction=0.85, val_fraction=0.15, test_fraction=0.0):
    base = OxfordIIITPet(root=root, split="trainval", target_types="segmentation",
                         download=True)
    # Fixed split is created once from the official trainval partition.
    train_idx, val_idx, _ = make_splits(
        len(base), seed, train_fraction, val_fraction, test_fraction
    )
    # The official test partition remains untouched by split randomization.
    official_test = OxfordIIITPet(root=root, split="test",
                                  target_types="segmentation", download=True)

    train_ds = PetSegmentationDataset(base, train_idx, image_size, augment_train)
    val_ds = PetSegmentationDataset(base, val_idx, image_size, augment=False)
    test_ds = PetSegmentationDataset(
        official_test, range(len(official_test)), image_size, augment=False
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    return DataBundle(train_loader, val_loader, test_loader, CLASS_NAMES)

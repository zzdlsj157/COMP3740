"""Image/mask pairs from a directory or a split manifest."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(directory):
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_TYPES)


def by_stem(directory):
    if not directory.is_dir():
        raise ValueError(f"Missing directory: {directory}")
    pairs = {}
    for path in image_paths(directory):
        stem = path.relative_to(directory).with_suffix("")
        if stem in pairs:
            raise ValueError(f"Duplicate stem in {directory}: {stem}")
        pairs[stem] = path
    return pairs


def read_pairs(source):
    source = Path(source).resolve()
    if source.is_dir():
        images = by_stem(source / "images")
        masks = by_stem(source / "masks")
        if images.keys() != masks.keys():
            missing = sorted(str(path) for path in images.keys() - masks.keys())
            extra = sorted(str(path) for path in masks.keys() - images.keys())
            raise ValueError(f"Unpaired files in {source}: missing masks={missing[:5]}, extra masks={extra[:5]}")
        pairs = [(images[stem].resolve(), masks[stem].resolve()) for stem in images]
    else:
        with source.open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            if not {"image", "mask"}.issubset(rows.fieldnames or []):
                raise ValueError(f"{source} must have image,mask columns.")
            pairs = []
            for line, row in enumerate(rows, start=2):
                if not row["image"] or not row["mask"]:
                    raise ValueError(f"{source}:{line} has an empty image or mask path.")
                pair = tuple((source.parent / row[column].strip()).resolve() for column in ("image", "mask"))
                for path in pair:
                    if not path.is_file():
                        raise ValueError(f"{source}:{line} references a missing file: {path}")
                pairs.append(pair)

    if not pairs:
        raise ValueError(f"No image/mask pairs found in {source}.")
    if len({image for image, _ in pairs}) != len(pairs):
        raise ValueError(f"{source} contains duplicate images.")
    return pairs


def split_train_val(train, val, fraction, seed):
    if val is not None:
        if {image for image, _ in train} & {image for image, _ in val}:
            raise ValueError("Training and validation contain the same image path.")
        return train, val
    if len(train) < 2 or not 0 < fraction < 1:
        raise ValueError("Automatic validation split needs at least two images and a fraction in (0, 1).")
    count = min(len(train) - 1, max(1, round(len(train) * fraction)))
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
    return [train[i] for i in order[count:]], [train[i] for i in order[:count]]


def read_image(path, size):
    with Image.open(path) as source:
        original_size = (source.height, source.width)
        resized = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        image = torch.from_numpy(np.array(resized, dtype=np.uint8))
    return image.permute(2, 0, 1).float().div_(255), original_size


class SmokeDataset(Dataset):
    def __init__(self, pairs, size, mask_threshold):
        self.pairs = pairs
        self.size = size
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image, original_size = read_image(image_path, self.size)
        with Image.open(mask_path) as source:
            if (source.height, source.width) != original_size:
                raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
            if source.mode not in ("1", "L", "P", "I", "I;16", "I;16B", "I;16L", "F"):
                source = source.convert("L")
            pixels = np.asarray(source)
            # Binarize before resizing; preserve class indices in palette masks.
            binary = Image.fromarray((pixels > self.mask_threshold).astype(np.uint8))
            resized = binary.resize((self.size, self.size), Image.Resampling.NEAREST)
            mask = torch.from_numpy(np.array(resized, dtype=np.uint8)).unsqueeze(0).float()
        return image, mask

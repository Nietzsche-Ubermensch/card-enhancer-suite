"""
Datasets for ELAN super-resolution training.

Ground-truth card images live in a folder; low-resolution inputs are
synthesized on the fly with bicubic downsampling (standard SR protocol).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def bicubic_downscale(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, w // scale), max(1, h // scale)), Image.BICUBIC)


class CardSRDataset(Dataset):
    """
    Random-crop training pairs from a folder of ground-truth card images.

    Each item is (lr, gt): a random gt_size x gt_size crop of the GT image
    and its bicubic-downscaled counterpart (gt_size // scale).
    """

    def __init__(
        self,
        root: Path,
        scale: int = 2,
        gt_size: int = 256,
        use_hflip: bool = True,
        use_rot: bool = True,
        repeat: int = 1,
    ):
        self.paths = list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"No training images found under {root}")
        self.scale = scale
        self.gt_size = gt_size
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.paths) * self.repeat

    def _load_gt(self, index: int) -> Image.Image:
        img = Image.open(self.paths[index % len(self.paths)]).convert("RGB")
        # Guarantee the crop fits: upscale small cards so they reach gt_size.
        w, h = img.size
        if w < self.gt_size or h < self.gt_size:
            factor = max(self.gt_size / w, self.gt_size / h)
            img = img.resize(
                (int(w * factor) + 1, int(h * factor) + 1), Image.BICUBIC
            )
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt = self._load_gt(index)
        w, h = gt.size
        x = random.randint(0, w - self.gt_size)
        y = random.randint(0, h - self.gt_size)
        gt = gt.crop((x, y, x + self.gt_size, y + self.gt_size))

        lr = bicubic_downscale(gt, self.scale)

        if self.use_hflip and random.random() < 0.5:
            gt, lr = TF.hflip(gt), TF.hflip(lr)
        if self.use_rot:
            k = random.randint(0, 3)
            if k:
                gt, lr = TF.rotate(gt, -90 * k), TF.rotate(lr, -90 * k)

        return TF.to_tensor(lr), TF.to_tensor(gt)


class CardSRValidDataset(Dataset):
    """Full-image validation pairs: (lr, gt, name)."""

    def __init__(self, root: Path, scale: int = 2):
        self.paths = list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"No validation images found under {root}")
        self.scale = scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[index]
        gt = Image.open(path).convert("RGB")
        # Round dimensions down to a multiple of scale so LR*scale == GT size.
        w, h = gt.size
        gt = gt.crop((0, 0, w - w % self.scale, h - h % self.scale))
        lr = bicubic_downscale(gt, self.scale)
        return TF.to_tensor(lr), TF.to_tensor(gt), path.name

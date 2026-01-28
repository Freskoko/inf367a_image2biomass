from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from main.utils.utils import DataType, VisionModelConfig


class ImagePathDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_paths: list[str],
        vision_cfg: VisionModelConfig,
        mode: DataType = DataType.VAL,
    ):
        self.root = root
        self.image_paths = image_paths
        self.mode = mode
        self.cfg = vision_cfg

        if self.mode == DataType.TRAIN:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_train_transform(self):
        # TODO MOVE ME SOMEWHERE ELSE!
        """
        When training across epochs,
        images are randomly flipped, either vertically, horizontally, or both
        Source: https://link.springer.com/article/10.1186/s40537-019-0197-0?
        """
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size * 2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.v2.GaussianNoise(mean = 1, std = 0.1), can hurt predictions
                transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std),
            ]
        )

    def _get_val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size * 2)),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg.mean, self.cfg.std),
            ]
        )

    def __getitem__(self, idx: int):
        rel = self.image_paths[idx]
        path = self.root / rel
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, rel


def _get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = torch.nn.Identity()
    m.eval()
    m.to(device)
    return m


@torch.inference_mode()
def extract_features(
    out_npy: Path,
    vision_cfg: VisionModelConfig,
    ds,
) -> np.ndarray:
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    device = _get_device(vision_cfg.device)
    dl = DataLoader(
        ds,
        batch_size=vision_cfg.batch_size,
        shuffle=False,
        num_workers=vision_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_feature_extractor(device)

    feats = []
    keys = []

    for xb, rels in dl:
        xb = xb.to(device, non_blocking=True)
        fb = model(xb).detach().cpu().numpy()
        feats.append(fb)
        keys.extend(rels)

    feats = np.concatenate(feats, axis=0)

    order = np.argsort(np.array(keys))
    feats = feats[order]
    keys = [keys[i] for i in order]

    np.save(out_npy, feats)
    (out_npy.with_suffix(".paths.txt")).write_text("\n".join(keys))

    return feats

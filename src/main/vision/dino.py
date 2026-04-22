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


def build_feature_extractor(
    device: torch.device,
    backbone: str = "dino",
    model_name: str = "dinov2_vits14",
):
    if backbone == "dino":
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)
        return model

    if backbone == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        return model

    if backbone == "convnext":
        from main.vision.ConvNeXt.convnext import (
            build_feature_extractor as build_convnext_feature_extractor,
        )

        return build_convnext_feature_extractor(device)

    raise ValueError(f"Unknown backbone: {backbone}")

@torch.inference_mode()
def extract_features(
    out_npy: Path,
    vision_cfg: VisionModelConfig,
    ds,
    backbone: str = "dino",
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

    model = build_feature_extractor(
        device=device,
        backbone=backbone,
    )

    feats = []
    keys = []

    for xb, rels in dl:
        xb = xb.to(device, non_blocking=True)
        fb = model(xb)

        if isinstance(fb, dict):
            fb = fb["x_norm_clstoken"]

        if fb.ndim != 2:
            raise ValueError(
                f"Expected 2-D feature tensor (batch, dim) from {backbone}, got shape {tuple(fb.shape)}."
            )

        feats.append(fb.cpu().numpy())
        keys.extend(rels)

    feats = np.concatenate(feats, axis=0)
    if feats.ndim != 2 or feats.shape[0] != len(keys):
        raise ValueError(
            f"Feature array shape {feats.shape} inconsistent with {len(keys)} image keys."
        )

    # we do not sort, we want feats in the mulitplied order they came in
    # order = np.argsort(np.array(keys))
    # feats = feats[order]
    # keys = [keys[i] for i in order]

    np.save(out_npy, feats)
    out_npy.with_suffix(".paths.txt").write_text("\n".join(keys))

    return feats

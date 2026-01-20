import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.cnn_model import CNNRegressor


TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


@dataclass(frozen=True)
class CNNConfig:
    train_wide_path: str = "../data/interim/train_wide.parquet"
    folds_path: str = "../data/interim/folds.parquet"
    images_root: str = "../data"
    out_dir: str = "../data/interim/cnn_models"

    backbone: str = "resnet18"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4

    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class BiomassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: Path, tfm, train: bool):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.tfm = tfm
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)

        if self.train:
            y = row[TARGETS].values.astype(np.float32)
            y = np.log1p(y)
            return img, torch.from_numpy(y)

        return img


def train_cnn(cfg: CNNConfig) -> Path:
    _set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.train_wide_path)
    folds = pd.read_parquet(cfg.folds_path)
    df = df.merge(folds, on="image_id", how="inner")

    tfm_train = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tfm_val = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    images_root = Path(cfg.images_root)

    all_ckpts = []

    for fold in sorted(df["fold"].unique()):
        tr_df = df[df["fold"] != fold].copy()
        va_df = df[df["fold"] == fold].copy()

        tr_ds = BiomassDataset(tr_df, images_root, tfm_train, train=True)
        va_ds = BiomassDataset(va_df, images_root, tfm_val, train=True)

        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)

        model = CNNRegressor(backbone=cfg.backbone, num_targets=len(TARGETS)).to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        best = float("inf")
        ckpt_path = out_dir / f"cnn_{cfg.backbone}_fold{fold}.pt"

        for epoch in range(cfg.epochs):
            model.train()
            tr_loss = 0.0

            for x, y in tr_ld:
                x = x.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()

                tr_loss += loss.item() * x.size(0)

            tr_loss /= len(tr_ds)

            model.eval()
            preds = []
            trues = []

            with torch.no_grad():
                for x, y in va_ld:
                    x = x.to(cfg.device, non_blocking=True)
                    y = y.numpy()
                    p = model(x).cpu().numpy()

                    preds.append(p)
                    trues.append(y)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            preds = np.expm1(preds)
            trues = np.expm1(trues)

            rmses = [_rmse(trues[:, i], preds[:, i]) for i in range(len(TARGETS))]
            avg = float(np.mean(rmses))

            print(f"Fold {fold} | Epoch {epoch+1:02d} | train_loss {tr_loss:.4f} | val_rmse {avg:.4f}")

            if avg < best:
                best = avg
                torch.save({"model": model.state_dict(), "best_rmse": best, "fold": fold}, ckpt_path)

        print(f"Fold {fold} best RMSE: {best:.4f}\n")
        all_ckpts.append(str(ckpt_path))

    meta_path = out_dir / f"checkpoints_{cfg.backbone}.txt"
    meta_path.write_text("\n".join(all_ckpts))
    return meta_path


if __name__ == "__main__":
    train_cnn(CNNConfig())

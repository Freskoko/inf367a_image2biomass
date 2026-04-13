from __future__ import annotations

import copy
import gc
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger(__name__)

BASE_TARGET_COLUMNS = ("Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g")
DERIVED_TARGET_COLUMNS = ("GDM_g", "Dry_Total_g")
ALL_TARGET_COLUMNS = BASE_TARGET_COLUMNS + DERIVED_TARGET_COLUMNS

TARGET_WEIGHTS = {
    "Dry_Clover_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Green_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

MODEL_REGISTRY = {
    "convnext_tiny": (
        models.convnext_tiny,
        models.ConvNeXt_Tiny_Weights.DEFAULT,
    ),
    "convnext_small": (
        models.convnext_small,
        models.ConvNeXt_Small_Weights.DEFAULT,
    ),
    "convnext_base": (
        models.convnext_base,
        models.ConvNeXt_Base_Weights.DEFAULT,
    ),
    "convnext_large": (
        models.convnext_large,
        models.ConvNeXt_Large_Weights.DEFAULT,
    ),
}


def configure_logging(level: str | int = logging.INFO) -> None:
    if isinstance(level, str):
        normalized_level = getattr(logging, level.upper(), logging.INFO)
    else:
        normalized_level = level

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=normalized_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(normalized_level)

    logger.setLevel(normalized_level)


def resolve_data_root(start: Path | None = None) -> Path:
    """Find the dataset directory without assuming a single project layout."""
    search_start = (start or Path.cwd()).resolve()

    for base in (search_start, *search_start.parents):
        for candidate in (base / "src" / "data", base / "data"):
            if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
                return candidate

    file_start = Path(__file__).resolve()
    for base in (file_start.parent, *file_start.parents):
        for candidate in (base / "src" / "data", base / "data"):
            if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
                return candidate

    raise FileNotFoundError(
        "Could not find a dataset directory containing both train.csv and test.csv."
    )


def resolve_output_dir() -> Path:
    """Store ConvNeXt artifacts next to the project's other model outputs."""
    return Path(__file__).resolve().parents[2] / "model_data" / "convnext"


def infer_device(device: str = "auto") -> torch.device:
    if device != "auto":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available on this machine.")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class ConvNeXtTrainingConfig:
    data_root: Path = field(default_factory=resolve_data_root)
    output_dir: Path = field(default_factory=resolve_output_dir)
    model_names: tuple[str, ...] = tuple(MODEL_REGISTRY.keys())
    image_size: int = 224
    batch_size: int = 8
    num_workers: int = 0
    n_splits: int = 5
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 2
    use_pretrained: bool = True
    device: str = "auto"
    random_state: int = 42
    clip_predictions_to_zero: bool = True
    use_amp: bool = True
    log_level: str = "INFO"


@dataclass
class FoldTrainingResult:
    best_state_dict: dict[str, torch.Tensor]
    best_val_loss: float
    best_epoch: int


@dataclass
class ModelArtifacts:
    model_name: str
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    submission: pd.DataFrame


def _build_collection_group(frame: pd.DataFrame) -> pd.Series:
    return frame[["Sampling_Date", "State", "Species"]].astype(str).agg(" | ".join, axis=1)


def load_training_frame(data_root: Path) -> pd.DataFrame:
    """Load one training row per image and keep all five targets for evaluation."""
    train_long = pd.read_csv(
        data_root / "train.csv",
        usecols=[
            "image_path",
            "Sampling_Date",
            "State",
            "Species",
            "target_name",
            "target",
        ],
    )

    metadata = train_long[
        ["image_path", "Sampling_Date", "State", "Species"]
    ].drop_duplicates(subset=["image_path"])

    target_frame = (
        train_long.pivot_table(
            index="image_path",
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    train_frame = metadata.merge(
        target_frame,
        on="image_path",
        how="inner",
        validate="one_to_one",
    )
    train_frame["image_relpath"] = train_frame["image_path"]
    train_frame["image_path"] = train_frame["image_path"].map(
        lambda rel_path: (data_root / rel_path).resolve()
    )
    train_frame["collection_group"] = _build_collection_group(train_frame)

    missing_targets = [col for col in ALL_TARGET_COLUMNS if col not in train_frame.columns]
    if missing_targets:
        raise ValueError(f"Missing expected targets in training data: {missing_targets}")

    return train_frame.sort_values("image_relpath").reset_index(drop=True)


def load_test_frame(data_root: Path) -> pd.DataFrame:
    """Load one test row per image so the image is scored only once."""
    test_frame = pd.read_csv(
        data_root / "test.csv",
        usecols=["sample_id", "image_path", "target_name"],
    )
    test_frame = test_frame.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    test_frame["image_relpath"] = test_frame["image_path"]
    test_frame["image_path"] = test_frame["image_path"].map(
        lambda rel_path: (data_root / rel_path).resolve()
    )
    return test_frame.sort_values("image_relpath").reset_index(drop=True)


def add_derived_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild the two dependent targets from the three independent ones.

    This keeps predictions biologically consistent:
    GDM_g = Dry_Green_g + Dry_Clover_g
    Dry_Total_g = Dry_Dead_g + GDM_g
    """
    out = frame.copy()
    out["GDM_g"] = out["Dry_Green_g"] + out["Dry_Clover_g"]
    out["Dry_Total_g"] = out["Dry_Dead_g"] + out["GDM_g"]
    return out


def build_submission_frame(data_root: Path, test_frame: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    sample_submission = pd.read_csv(data_root / "sample_submission.csv")

    pred_long = (
        pd.concat([test_frame[["image_relpath"]].reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        .melt(
            id_vars=["image_relpath"],
            value_vars=list(ALL_TARGET_COLUMNS),
            var_name="target_name",
            value_name="target",
        )
    )
    pred_long["sample_id"] = (
        pred_long["image_relpath"].map(lambda path: Path(path).stem) + "__" + pred_long["target_name"]
    )

    submission = sample_submission[["sample_id"]].merge(
        pred_long[["sample_id", "target"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if submission["target"].isna().any():
        missing = int(submission["target"].isna().sum())
        raise ValueError(f"Submission is missing {missing} predictions after the merge.")

    return submission


class BiomassImageDataset(Dataset):
    """Return an RGB image tensor and, when available, its target vector."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        targets: np.ndarray | None,
        transform: transforms.Compose,
    ):
        self.image_paths = [Path(path) for path in image_paths]
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        if self.targets is None:
            return image_tensor, str(image_path)

        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image_tensor, target_tensor


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def build_dataloader(
    image_paths: Sequence[Path],
    targets: np.ndarray | None,
    transform: transforms.Compose,
    config: ConvNeXtTrainingConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = BiomassImageDataset(image_paths=image_paths, targets=targets, transform=transform)
    resolved_device = infer_device(config.device)
    use_cuda = resolved_device.type == "cuda"

    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        persistent_workers=config.num_workers > 0,
    )


def get_autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_convnext_regressor(
    model_name: str,
    output_dim: int,
    use_pretrained: bool,
    device: torch.device,
) -> nn.Module:
    """Create a ConvNeXt model and replace the classification head with regression outputs."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    builder, pretrained_weights = MODEL_REGISTRY[model_name]
    weights = pretrained_weights if use_pretrained else None
    model = builder(weights=weights)

    # We swap the final classification layer for regression outputs.
    # The backbone stays trainable, so the ImageNet weights are fine-tuned end to end.
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, output_dim)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    return model.to(device)


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: ConvNeXtTrainingConfig,
) -> FoldTrainingResult:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
    )
    use_amp = device.type == "cuda" and config.use_amp
    grad_scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad()
            with get_autocast_context(device=device, use_amp=config.use_amp):
                predictions = model(images)
                loss = criterion(predictions, targets)

            grad_scaler.scale(loss).backward()

            # Gradient clipping keeps unusually large updates from destabilizing fine-tuning.
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                targets = targets.to(device, non_blocking=device.type == "cuda")

                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)

                with get_autocast_context(device=device, use_amp=config.use_amp):
                    predictions = model(images)
                    val_loss += criterion(predictions, targets).item()

        current_lr = optimizer.param_groups[0]["lr"]
        val_loss /= max(len(val_loader), 1)
        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | lr=%.6f",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            current_lr,
        )
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            logger.info("New best validation loss at epoch %s: %.4f", epoch, val_loss)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(
                "Early stopping triggered at epoch %s after %s epochs without improvement.",
                epoch,
                config.patience,
            )
            break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = config.epochs

    return FoldTrainingResult(
        best_state_dict=best_state_dict,
        best_val_loss=float(best_val_loss),
        best_epoch=best_epoch,
    )


@torch.inference_mode()
def predict_scaled_targets(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    preds: list[np.ndarray] = []

    model.eval()
    for images, _ in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with get_autocast_context(device=device, use_amp=use_amp):
            batch_preds = model(images).detach().float().cpu().numpy()
        preds.append(batch_preds)

    return np.concatenate(preds, axis=0)


def inverse_scale_predictions(
    scaled_preds: np.ndarray,
    scaler: StandardScaler,
    clip_to_zero: bool,
) -> pd.DataFrame:
    preds = scaler.inverse_transform(scaled_preds)
    if clip_to_zero:
        preds = np.clip(preds, a_min=0.0, a_max=None)

    pred_frame = pd.DataFrame(preds, columns=BASE_TARGET_COLUMNS)
    return add_derived_targets(pred_frame)


def weighted_r2_global(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    y_true_vec = np.concatenate([y_true[col].to_numpy() for col in ALL_TARGET_COLUMNS], axis=0)
    y_pred_vec = np.concatenate([y_pred[col].to_numpy() for col in ALL_TARGET_COLUMNS], axis=0)
    weights = np.concatenate(
        [
            np.full(len(y_true), TARGET_WEIGHTS[col], dtype=float)
            for col in ALL_TARGET_COLUMNS
        ],
        axis=0,
    )

    weighted_mean = np.sum(weights * y_true_vec) / np.sum(weights)
    ss_res = np.sum(weights * (y_true_vec - y_pred_vec) ** 2)
    ss_tot = np.sum(weights * (y_true_vec - weighted_mean) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {
        "weighted_r2": weighted_r2_global(y_true, y_pred),
    }

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for col in ALL_TARGET_COLUMNS:
        y_true_col = y_true[col].to_numpy()
        y_pred_col = y_pred[col].to_numpy()

        mae = float(mean_absolute_error(y_true_col, y_pred_col))
        rmse = float(np.sqrt(np.mean((y_true_col - y_pred_col) ** 2)))
        r2 = float(r2_score(y_true_col, y_pred_col))

        metrics[f"{col}_mae"] = mae
        metrics[f"{col}_rmse"] = rmse
        metrics[f"{col}_r2"] = r2

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    metrics["mean_mae"] = float(np.mean(mae_scores))
    metrics["mean_rmse"] = float(np.mean(rmse_scores))
    metrics["mean_r2"] = float(np.mean(r2_scores))
    return metrics


def make_oof_frame(train_frame: pd.DataFrame, oof_predictions: np.ndarray) -> pd.DataFrame:
    pred_frame = pd.DataFrame(oof_predictions, columns=ALL_TARGET_COLUMNS)

    out = train_frame[["image_relpath", "Sampling_Date", "State", "Species", "collection_group"]].copy()
    for col in ALL_TARGET_COLUMNS:
        out[f"true_{col}"] = train_frame[col].to_numpy()
        out[f"pred_{col}"] = pred_frame[col].to_numpy()

    return out


def cross_validate_model(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: ConvNeXtTrainingConfig,
    model_name: str,
) -> ModelArtifacts:
    configure_logging(config.log_level)
    device = infer_device(config.device)
    logger.info(
        "Starting cross-validation for %s on %s | folds=%s | batch_size=%s | epochs=%s | pretrained=%s | amp=%s",
        model_name,
        device,
        config.n_splits,
        config.batch_size,
        config.epochs,
        config.use_pretrained,
        device.type == "cuda" and config.use_amp,
    )

    if train_frame["collection_group"].nunique() < config.n_splits:
        raise ValueError(
            "Not enough unique collection groups for leakage-aware cross-validation."
        )

    train_transform, eval_transform = build_transforms(config.image_size)
    splitter = GroupKFold(n_splits=config.n_splits)

    oof_predictions = np.zeros((len(train_frame), len(ALL_TARGET_COLUMNS)), dtype=np.float32)
    fold_test_predictions: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []

    test_loader = build_dataloader(
        image_paths=test_frame["image_path"].tolist(),
        targets=None,
        transform=eval_transform,
        config=config,
        shuffle=False,
    )

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(train_frame, groups=train_frame["collection_group"]),
        start=1,
    ):
        logger.info("Fold %s/%s for %s", fold, config.n_splits, model_name)
        train_split = train_frame.iloc[train_idx].reset_index(drop=True)
        val_split = train_frame.iloc[val_idx].reset_index(drop=True)

        overlap = set(train_split["collection_group"]) & set(val_split["collection_group"])
        if overlap:
            raise ValueError(
                f"Leakage detected in fold {fold}: train and validation share collection groups."
            )

        logger.info(
            "Fold %s split | train_images=%s | val_images=%s | train_groups=%s | val_groups=%s",
            fold,
            len(train_split),
            len(val_split),
            train_split["collection_group"].nunique(),
            val_split["collection_group"].nunique(),
        )

        scaler = StandardScaler().fit(train_split[list(BASE_TARGET_COLUMNS)])
        train_targets = scaler.transform(train_split[list(BASE_TARGET_COLUMNS)])
        val_targets = scaler.transform(val_split[list(BASE_TARGET_COLUMNS)])

        train_loader = build_dataloader(
            image_paths=train_split["image_path"].tolist(),
            targets=train_targets,
            transform=train_transform,
            config=config,
            shuffle=True,
        )
        val_loader = build_dataloader(
            image_paths=val_split["image_path"].tolist(),
            targets=val_targets,
            transform=eval_transform,
            config=config,
            shuffle=False,
        )

        model = build_convnext_regressor(
            model_name=model_name,
            output_dim=len(BASE_TARGET_COLUMNS),
            use_pretrained=config.use_pretrained,
            device=device,
        )
        training_result = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
        )
        model.load_state_dict(training_result.best_state_dict)

        val_scaled_preds = predict_scaled_targets(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=config.use_amp,
        )
        val_preds = inverse_scale_predictions(
            scaled_preds=val_scaled_preds,
            scaler=scaler,
            clip_to_zero=config.clip_predictions_to_zero,
        )
        val_truth = val_split[list(ALL_TARGET_COLUMNS)].reset_index(drop=True)

        test_scaled_preds = predict_scaled_targets(
            model=model,
            data_loader=test_loader,
            device=device,
            use_amp=config.use_amp,
        )
        test_preds = inverse_scale_predictions(
            scaled_preds=test_scaled_preds,
            scaler=scaler,
            clip_to_zero=config.clip_predictions_to_zero,
        )

        oof_predictions[val_idx] = val_preds[list(ALL_TARGET_COLUMNS)].to_numpy(dtype=np.float32)
        fold_test_predictions.append(test_preds[list(ALL_TARGET_COLUMNS)].to_numpy(dtype=np.float32))

        metrics = evaluate_predictions(y_true=val_truth, y_pred=val_preds)
        metrics.update(
            {
                "model_name": model_name,
                "fold": fold,
                "best_epoch": training_result.best_epoch,
                "best_val_loss": training_result.best_val_loss,
                "n_train_images": int(len(train_split)),
                "n_val_images": int(len(val_split)),
                "n_train_groups": int(train_split["collection_group"].nunique()),
                "n_val_groups": int(val_split["collection_group"].nunique()),
            }
        )
        fold_metrics.append(metrics)

        logger.info(
            "Fold %s complete | weighted_r2=%.4f | mean_r2=%.4f | mean_mae=%.4f | best_epoch=%s | best_val_loss=%.4f",
            fold,
            metrics["weighted_r2"],
            metrics["mean_r2"],
            metrics["mean_mae"],
            training_result.best_epoch,
            training_result.best_val_loss,
        )

        del model
        clear_device_cache(device)

    averaged_test_preds = np.mean(np.stack(fold_test_predictions, axis=0), axis=0)
    test_pred_frame = pd.DataFrame(averaged_test_preds, columns=ALL_TARGET_COLUMNS)
    test_pred_frame.insert(0, "image_relpath", test_frame["image_relpath"].to_numpy())

    oof_frame = make_oof_frame(train_frame=train_frame, oof_predictions=oof_predictions)
    submission = build_submission_frame(
        data_root=config.data_root,
        test_frame=test_frame,
        preds=test_pred_frame[list(ALL_TARGET_COLUMNS)],
    )

    return ModelArtifacts(
        model_name=model_name,
        fold_metrics=pd.DataFrame(fold_metrics),
        oof_predictions=oof_frame,
        test_predictions=test_pred_frame,
        submission=submission,
    )


def run_convnext_experiment(
    config: ConvNeXtTrainingConfig,
) -> tuple[pd.DataFrame, dict[str, ModelArtifacts]]:
    configure_logging(config.log_level)
    seed_everything(config.random_state)
    device = infer_device(config.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        device_index = 0 if device.index is None else device.index
        total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        logger.info(
            "Using CUDA device %s (%s) with %.2f GB VRAM",
            device_index,
            torch.cuda.get_device_name(device_index),
            total_memory_gb,
        )
    else:
        logger.info("Using device: %s", device)

    train_frame = load_training_frame(config.data_root)
    test_frame = load_test_frame(config.data_root)
    logger.info(
        "Loaded data | train_images=%s | test_images=%s | collection_groups=%s",
        len(train_frame),
        len(test_frame),
        train_frame["collection_group"].nunique(),
    )

    results: dict[str, ModelArtifacts] = {}
    summary_rows: list[dict[str, float | str]] = []

    for model_name in config.model_names:
        logger.info("Evaluating model: %s", model_name)
        artifacts = cross_validate_model(
            train_frame=train_frame,
            test_frame=test_frame,
            config=config,
            model_name=model_name,
        )
        results[model_name] = artifacts

        fold_metrics = artifacts.fold_metrics
        summary_row: dict[str, float | str] = {
            "model_name": model_name,
            "mean_weighted_r2": float(fold_metrics["weighted_r2"].mean()),
            "std_weighted_r2": float(fold_metrics["weighted_r2"].std(ddof=0)),
            "mean_mae": float(fold_metrics["mean_mae"].mean()),
            "mean_rmse": float(fold_metrics["mean_rmse"].mean()),
            "mean_r2": float(fold_metrics["mean_r2"].mean()),
        }

        for target_col in ALL_TARGET_COLUMNS:
            summary_row[f"mean_{target_col}_r2"] = float(fold_metrics[f"{target_col}_r2"].mean())

        summary_rows.append(summary_row)
        logger.info(
            "Model %s summary | mean_weighted_r2=%.4f | mean_r2=%.4f | mean_mae=%.4f",
            model_name,
            summary_row["mean_weighted_r2"],
            summary_row["mean_r2"],
            summary_row["mean_mae"],
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        "mean_weighted_r2",
        ascending=False,
    )
    logger.info("ConvNeXt experiment complete.")
    return summary_frame.reset_index(drop=True), results


def save_experiment_outputs(
    summary_frame: pd.DataFrame,
    results: dict[str, ModelArtifacts],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_dir / "convnext_cv_summary.csv", index=False)
    logger.info("Saved summary metrics to %s", output_dir / "convnext_cv_summary.csv")

    for model_name, artifacts in results.items():
        artifacts.fold_metrics.to_csv(output_dir / f"{model_name}_fold_metrics.csv", index=False)
        artifacts.oof_predictions.to_csv(output_dir / f"{model_name}_oof_predictions.csv", index=False)
        artifacts.test_predictions.to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)
        artifacts.submission.to_csv(output_dir / f"{model_name}_submission.csv", index=False)
        logger.info("Saved model artifacts for %s to %s", model_name, output_dir)


def main() -> None:
    config = ConvNeXtTrainingConfig()
    summary_frame, results = run_convnext_experiment(config)
    save_experiment_outputs(summary_frame=summary_frame, results=results, output_dir=config.output_dir)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()

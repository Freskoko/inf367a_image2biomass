import copy
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_green_mass_frame(data_root: Path) -> pd.DataFrame:
    train_df = pd.read_csv(
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

    green_mass_df = train_df.loc[
        train_df["target_name"] == "GDM_g",
        ["image_path", "Sampling_Date", "State", "Species", "target"],
    ].drop_duplicates(subset=["image_path"])

    green_mass_df = green_mass_df.reset_index(drop=True)
    green_mass_df["image_path"] = green_mass_df["image_path"].map(
        lambda rel_path: data_root / rel_path
    )
    green_mass_df["collection_group"] = green_mass_df[
        ["Sampling_Date", "State", "Species"]
    ].astype(str).agg(" | ".join, axis=1)
    return green_mass_df


class BiomasDataset(Dataset):
    """Custom dataset for images and green mass labels."""

    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.float32)


class ConvNeXtRegressor:
    """ConvNeXt model for green mass regression using cross-validation."""

    def __init__(
        self,
        model_name: str = "convnext_base",
        device: str | None = None,
        pretrained: bool = True,
        random_state: int = 42,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self.random_state = random_state
        self.model = self._load_model()

    def _load_model(self):
        """Load pre-trained ConvNeXt model and modify for regression."""
        logger.info(f"Loading {self.model_name} from PyTorch torchvision...")

        weights = "DEFAULT" if self.pretrained else None
        model = (
            models.convnext_base(weights=weights)
            if self.model_name == "convnext_base"
            else models.convnext_small(weights=weights)
            if self.model_name == "convnext_small"
            else models.convnext_large(weights=weights)
            if self.model_name == "convnext_large"
            else models.convnext_tiny(weights=weights)
        )

        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, 1)

        model = model.to(self.device)
        return model

    def train_fold(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
    ):
        """Train model on a single fold."""
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return best_val_loss

    def evaluate_loader(self, data_loader, target_scaler: StandardScaler) -> dict:
        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for images, batch_labels in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions.extend(outputs.cpu().numpy().flatten().tolist())
                labels.extend(batch_labels.numpy().flatten().tolist())

        y_true = target_scaler.inverse_transform(
            np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        ).flatten()
        y_pred = target_scaler.inverse_transform(
            np.asarray(predictions, dtype=np.float32).reshape(-1, 1)
        ).flatten()

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)),
        }

    def cross_validate(
        self,
        data_df: pd.DataFrame,
        n_splits: int = 5,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """Train using leakage-aware cross-validation."""
        logger.info(f"Starting {n_splits}-fold cross-validation...")

        train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        data_df = data_df.sample(frac=1.0, random_state=self.random_state).reset_index(
            drop=True
        )

        if group_col is not None and data_df[group_col].nunique() < n_splits:
            raise ValueError(
                f"Not enough unique groups in {group_col} "
                f"for {n_splits}-fold cross-validation."
            )

        if group_col is None:
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            split_iterator = splitter.split(data_df)
            logger.info("Using shuffled KFold on images.")
        else:
            splitter = GroupKFold(n_splits=n_splits)
            split_iterator = splitter.split(data_df, groups=data_df[group_col])
            logger.info(f"Using GroupKFold grouped by {group_col}.")

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(split_iterator, start=1):
            logger.info(f"\nFold {fold}/{n_splits}")

            train_df = data_df.iloc[train_idx].reset_index(drop=True)
            val_df = data_df.iloc[val_idx].reset_index(drop=True)

            if group_col is not None:
                overlap = set(train_df[group_col]) & set(val_df[group_col])
                if overlap:
                    raise ValueError(
                        f"Leakage detected in fold {fold}: "
                        f"{len(overlap)} shared groups across train and val."
                    )

            target_scaler = StandardScaler().fit(train_df[["target"]])
            train_labels = target_scaler.transform(train_df[["target"]]).flatten()
            val_labels = target_scaler.transform(val_df[["target"]]).flatten()

            train_dataset = BiomasDataset(
                train_df["image_path"].tolist(), train_labels, train_transforms
            )
            val_dataset = BiomasDataset(
                val_df["image_path"].tolist(), val_labels, val_transforms
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=min(batch_size, len(train_dataset)),
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(batch_size, len(val_dataset)),
                shuffle=False,
            )

            self.model = self._load_model()

            best_val_loss = self.train_fold(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
            )

            metrics = self.evaluate_loader(val_loader, target_scaler)
            metrics.update(
                {
                    "fold": fold,
                    "best_val_loss": float(best_val_loss),
                    "n_train": int(len(train_df)),
                    "n_val": int(len(val_df)),
                    "n_train_groups": int(train_df[group_col].nunique())
                    if group_col is not None
                    else int(train_df["image_path"].nunique()),
                    "n_val_groups": int(val_df[group_col].nunique())
                    if group_col is not None
                    else int(val_df["image_path"].nunique()),
                }
            )
            fold_metrics.append(metrics)

            logger.info(
                f"Fold {fold} - "
                f"best_val_loss={metrics['best_val_loss']:.4f}, "
                f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                f"R2={metrics['r2']:.3f}"
            )

        results_df = pd.DataFrame(fold_metrics)
        summary = results_df[["mae", "rmse", "r2"]].agg(["mean", "std"])

        logger.info("\nCross-validation complete")
        logger.info(f"Mean MAE: {summary.loc['mean', 'mae']:.3f}")
        logger.info(f"Mean RMSE: {summary.loc['mean', 'rmse']:.3f}")
        logger.info(f"Mean R2: {summary.loc['mean', 'r2']:.3f}")
        logger.info(f"R2 std: {summary.loc['std', 'r2']:.3f}")

        return results_df

    def predict(
        self, image_paths, target_scaler: StandardScaler, batch_size: int = 32
    ):
        """Make predictions on new images."""
        val_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = BiomasDataset(image_paths, [0] * len(image_paths), val_transforms)
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions.extend(outputs.cpu().numpy().flatten().tolist())

        predictions = np.array(predictions, dtype=np.float32).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()

        return predictions


def main():
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_models = (
        "convnext_tiny,convnext_small,convnext_base"
        if device == "cuda"
        else "convnext_tiny"
    )

    model_names = [
        name.strip()
        for name in os.getenv("CONVNEXT_MODELS", default_models).split(",")
        if name.strip()
    ]
    n_splits = int(os.getenv("CONVNEXT_N_SPLITS", "5"))
    batch_size = int(os.getenv("CONVNEXT_BATCH_SIZE", "8" if device == "cpu" else "32"))
    epochs = int(os.getenv("CONVNEXT_EPOCHS", "3" if device == "cpu" else "20"))
    lr = float(os.getenv("CONVNEXT_LR", "1e-4"))
    weight_decay = float(os.getenv("CONVNEXT_WEIGHT_DECAY", "1e-4"))
    pretrained = os.getenv("CONVNEXT_PRETRAINED", "1") != "0"
    group_mode = os.getenv("CONVNEXT_GROUP_MODE", "collection")

    if group_mode not in {"image", "collection"}:
        raise ValueError("CONVNEXT_GROUP_MODE must be either 'image' or 'collection'.")

    group_col = None if group_mode == "image" else "collection_group"

    data_root = Path(__file__).resolve().parents[3] / "data"
    data_df = build_green_mass_frame(data_root)

    logger.info("ConvNeXt Green Mass Regression")
    logger.info(f"Device: {device}")
    logger.info(f"Models to evaluate: {model_names}")
    logger.info(f"Rows: {len(data_df)}, unique images: {data_df['image_path'].nunique()}")
    logger.info(f"Group mode: {group_mode}")

    for model_name in model_names:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating {model_name}")
        logger.info(f"{'=' * 50}")

        regressor = ConvNeXtRegressor(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            random_state=42,
        )
        results_df = regressor.cross_validate(
            data_df=data_df,
            n_splits=n_splits,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            group_col=group_col,
        )

        out_path = (
            Path(__file__).resolve().parent
            / f"{model_name}_{group_mode}_cv_results.csv"
        )
        results_df.to_csv(out_path, index=False)
        logger.info(f"Saved fold metrics to {out_path}")


if __name__ == "__main__":
    main()

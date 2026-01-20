from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


TARGETS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


@dataclass(frozen=True)
class PreprocessConfig:
    train_csv: str
    test_csv: str
    out_dir: str = "data/interim"
    n_folds: int = 5


def _make_image_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["image_id"] = (
        df["image_path"]
        .str.split("/")
        .str[-1]
        .str.replace(".jpg", "", regex=False)
    )
    return df


def _date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Sampling_Date" not in df.columns:
        return df
    
    dt = pd.to_datetime(df["Sampling_Date"], errors="coerce")

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month

    angle = 2 * np.pi * df["month"] / 12.0
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)

    return df


def _pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    potential_id_cols = [
        "image_id",
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "year",
        "month",
        "month_sin",
        "month_cos",
    ]
    
    id_cols = [col for col in potential_id_cols if col in df.columns]

    wide = (
        df.pivot_table(
            index=id_cols,
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
    )

    return wide


def make_wide_tables(cfg: PreprocessConfig) -> tuple[Path, Path]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(cfg.train_csv)
    test = pd.read_csv(cfg.test_csv)

    train = _make_image_id(train)
    test = _make_image_id(test)

    train = _date_features(train)
    test = _date_features(test)

    train_wide = _pivot_wide(train)
    test_wide = test.drop_duplicates("image_id").reset_index(drop=True)

    train_path = out_dir / "train_wide.parquet"
    test_path = out_dir / "test_wide.parquet"

    train_wide.to_parquet(train_path, index=False)
    test_wide.to_parquet(test_path, index=False)

    return train_path, test_path


def make_folds(train_wide_path: Path, cfg: PreprocessConfig) -> Path:
    df = pd.read_parquet(train_wide_path)

    gkf = GroupKFold(n_splits=cfg.n_folds)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        gkf.split(df, groups=df["image_id"])
    ):
        df.loc[val_idx, "fold"] = fold

    folds_path = Path(cfg.out_dir) / "folds.parquet"
    df[["image_id", "fold"]].to_parquet(folds_path, index=False)

    return folds_path




if __name__ == "__main__":
    cfg = PreprocessConfig(
        train_csv="data/train.csv",
        test_csv="data/test.csv",
        out_dir="data/interim",
        n_folds=5,
    )

    train_wide_path, test_wide_path = make_wide_tables(cfg)
    folds_path = make_folds(train_wide_path, cfg)

    train = pd.read_parquet(train_wide_path)
    folds = pd.read_parquet(folds_path)

    print("Train wide shape:", train.shape)
    print("Unique images:", train["image_id"].nunique())
    print("Target columns present:",
          all(t in train.columns for t in TARGETS))
    print("Fold distribution:")
    print(folds["fold"].value_counts().sort_index())

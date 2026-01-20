from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path = Path(__file__).parent.parent.parent / "data"

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.root / "test.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _add_date_features(df: pd.DataFrame, col: str = "Sampling_Date") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[col], errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(out[col], format="%Y/%m/%d", errors="coerce")

    out["year"] = dt.dt.year.astype("Int64")
    out["month"] = dt.dt.month.astype("Int64")
    out["day"] = dt.dt.day.astype("Int64")
    out["dayofyear"] = dt.dt.dayofyear.astype("Int64")
    return out


def pivot_train_long_to_wide(df_long: pd.DataFrame, targets: Iterable[str] = TARGETS) -> pd.DataFrame:
    id_cols = [
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]

    df = df_long[id_cols + ["target_name", "target"]].copy()

    wide = (
        df.pivot_table(
            index=id_cols,
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
    )

    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan

    return wide


def make_features_train(df_wide: pd.DataFrame, targets: Iterable[str] = TARGETS):
    df = _add_date_features(df_wide)

    feature_cols = [
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "State",
        "Species",
        "year",
        "month",
        "day",
        "dayofyear",
    ]

    X = df[feature_cols].copy()
    y = df[list(targets)].copy()
    groups = df["image_path"].to_numpy()
    meta = df[["image_path"]].copy()
    return X, y, groups, meta


def make_features_test(df_test: pd.DataFrame):
    if "Sampling_Date" in df_test.columns:
        df = _add_date_features(df_test)
    else:
        df = df_test.copy()

    feature_cols = [
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "State",
        "Species",
        "year",
        "month",
        "day",
        "dayofyear",
    ]

    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].copy() if available_cols else pd.DataFrame()
    groups = df["image_path"].to_numpy()
    meta = df[["image_path"]].copy()
    return X, groups, meta

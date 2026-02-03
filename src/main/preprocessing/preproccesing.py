from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pivot_train_long_to_wide(
    df_long: pd.DataFrame, targets: Iterable[str]
) -> pd.DataFrame:
    id_cols = [
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]

    df = df_long[id_cols + ["target_name", "target"]].copy()

    wide = df.pivot_table(
        index=id_cols,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan

    return wide


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

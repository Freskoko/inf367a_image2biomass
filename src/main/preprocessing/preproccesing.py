from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

    # Guard against silent data loss: pivot_table(aggfunc="first") would hide
    # duplicate (image_path, target_name) rows. Surface them instead.
    dup_mask = df.duplicated(subset=["image_path", "target_name"], keep=False)
    if dup_mask.any():
        n_dups = int(dup_mask.sum())
        raise ValueError(
            f"train.csv has {n_dups} duplicate (image_path, target_name) rows."
        )

    wide = df.pivot_table(
        index=id_cols,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan

    # Warn (don't raise) when composite targets disagree with the component
    # sum. CV recomputes them from the base targets, so any mismatched row
    # scores against reconstructed ground truth instead of what's in the CSV.
    if {"Dry_Green_g", "Dry_Clover_g", "GDM_g"}.issubset(wide.columns):
        gdm_check = wide["Dry_Green_g"].fillna(0) + wide["Dry_Clover_g"].fillna(0)
        gdm_diff = (wide["GDM_g"].fillna(gdm_check) - gdm_check).abs()
        n_gdm = int((gdm_diff > 1e-3).sum())
        if n_gdm > 0:
            warnings.warn(
                f"{n_gdm} row(s) have GDM_g != Dry_Green_g + Dry_Clover_g "
                f"(max diff {gdm_diff.max():.4f}); CV recomputes from components.",
                stacklevel=2,
            )
    if {
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "Dry_Total_g",
    }.issubset(wide.columns):
        total_check = (
            wide["Dry_Green_g"].fillna(0)
            + wide["Dry_Dead_g"].fillna(0)
            + wide["Dry_Clover_g"].fillna(0)
        )
        total_diff = (wide["Dry_Total_g"].fillna(total_check) - total_check).abs()
        n_total = int((total_diff > 1e-3).sum())
        if n_total > 0:
            warnings.warn(
                f"{n_total} row(s) have Dry_Total_g != Green + Dead + Clover "
                f"(max diff {total_diff.max():.4f}); CV recomputes from components.",
                stacklevel=2,
            )

    return wide


def make_features_train(df_wide: pd.DataFrame, targets: Iterable[str]):
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

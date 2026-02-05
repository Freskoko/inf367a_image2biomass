import pandas as pd


def merge_features(X_meta: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    X = X_meta.copy()
    X = X.merge(feature_df, on="image_path", how="left", validate="one_to_one")

    missing = X.isna().any(axis=1).sum()
    if missing:
        raise ValueError(
            f"Missing ResNet features for {missing} rows. Check paths alignment."
        )

    X.columns = X.columns.astype(str)
    return X

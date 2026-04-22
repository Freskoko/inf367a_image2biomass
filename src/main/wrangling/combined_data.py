import pandas as pd

def merge_features(X_meta: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    X = X_meta.copy()

    # instance id create
    feature_df = feature_df.copy()
    feature_df["instance_id"] = feature_df.index.astype(str)

    # instance_id, no longer the the image_path
    X = X.merge(feature_df, on="instance_id", how="left", validate="one_to_one")

    # remove old id
    if "instance_id" in X.columns:
        X = X.drop(columns=["instance_id"])

    X.columns = X.columns.astype(str)
    return X
import pandas as pd

def merge_features(X_meta: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    X = X_meta.copy()
    
    # 1. Create a unique ID for the features based on their position (0, 1, 2...)
    # This matches the 'instance_id' we created in load_data
    feature_df = feature_df.copy()
    feature_df["instance_id"] = feature_df.index.astype(str)
    
    # 2. Merge on the unique instance_id, NOT the image_path
    # We can keep validate="one_to_one" because instance_id is unique!
    X = X.merge(feature_df, on="instance_id", how="left", validate="one_to_one")

    missing = X.isna().any(axis=1).sum()
    if missing:
        raise ValueError(
            f"Missing features for {missing} rows. Did you forget to delete the old .npy cache?"
        )

    # 3. Clean up: remove the ID and the raw image_path so the model only sees numbers
    if "instance_id" in X.columns:
        X = X.drop(columns=["instance_id"])
    
    X.columns = X.columns.astype(str)
    return X
from __future__ import annotations

from pathlib import Path

from preproccesing import DatasetPaths, read_csv, pivot_train_long_to_wide, make_features_train_with_id
from rf_regressor import RFConfig, load_feature_store, merge_features, cv_mean_r2


def main():
    paths = DatasetPaths()

    df_long = read_csv(paths.train_csv)
    df_wide = pivot_train_long_to_wide(df_long)

    X_meta, y, groups = make_features_train_with_id(df_wide)

    feature_df = load_feature_store(Path(__file__).parent / "model_data" / "features_train.npy")
    X = merge_features(X_meta, feature_df)

    cfg = RFConfig(n_splits=5, random_state=42)
    res = cv_mean_r2(cfg, X, y, groups)

    print("\nCV mean R2:", res["mean_r2"])
    print("Per-target R2:", res["per_target_r2"])


if __name__ == "__main__":
    main()

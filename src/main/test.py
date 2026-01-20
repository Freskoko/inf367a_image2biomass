from __future__ import annotations

from pathlib import Path

from preproccesing import DatasetPaths, read_csv
from resnet import ResnetConfig, extract_features


def main():
    paths = DatasetPaths()
    df_train = read_csv(paths.train_csv)

    images_root = paths.root
    out_dir = Path(__file__).parent / "model_data"
    out_npy = out_dir / "features_train.npy"

    cfg = ResnetConfig(batch_size=32, device="auto")
    feats = extract_features(df_train, images_root=images_root, out_npy=out_npy, cfg=cfg)

    print("features:", feats.shape)
    print("saved:", out_npy)


if __name__ == "__main__":
    main()

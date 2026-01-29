import pandas as pd

from main.utils.utils import DatasetPaths


def save_predictions(path_cfg: DatasetPaths, long_pred: pd.DataFrame) -> None:
    sample_sub = pd.read_csv(path_cfg.root / "sample_submission.csv")
    sub = sample_sub.merge(
        long_pred,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if sub["pred"].isna().any():
        missing = int(sub["pred"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions after merge.")

    out_path = path_cfg.root / "submission.csv"

    sub = sub.drop(columns=["target"]).rename(columns={"pred": "target"})
    sub.to_csv(out_path, index=False)

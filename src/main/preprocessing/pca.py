from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from main.utils.utils import TrainConfig


def apply_pca_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _embedding_cols(X: pd.DataFrame) -> list:
        return [
            c
            for c in X.columns
            if (isinstance(c, int)) or (isinstance(c, str) and c.isdigit())
        ]

    emb_cols = _embedding_cols(X_train)
    # order on train/test
    emb_cols = sorted(emb_cols, key=lambda x: int(x))

    Ztr = X_train[emb_cols].to_numpy(dtype=np.float32)
    Zte = X_test[emb_cols].to_numpy(dtype=np.float32)

    pca = PCA(
        n_components=train_cfg.pca_n_components,
        random_state=train_cfg.random_state,
    ).fit(Ztr)

    Ztr_p = pca.transform(Ztr)
    Zte_p = pca.transform(Zte)

    Xtr2 = X_train.drop(columns=emb_cols).copy()
    Xte2 = X_test.drop(columns=emb_cols).copy()

    cols = [f"pca_{i}" for i in range(train_cfg.pca_n_components)]
    Xtr_p = pd.DataFrame(Ztr_p, columns=cols, index=X_train.index)
    Xte_p = pd.DataFrame(Zte_p, columns=cols, index=X_test.index)

    Xtr_final = pd.concat([Xtr2, Xtr_p], axis=1)
    Xte_final = pd.concat([Xte2, Xte_p], axis=1)

    return Xtr_final, Xte_final

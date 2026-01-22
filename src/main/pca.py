from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAConfig:
    n_components: int = 128
    whiten: bool = False
    random_state: int = 42


def fit_pca(X: np.ndarray, cfg: PCAConfig) -> PCA:
    pca = PCA(n_components=cfg.n_components, whiten=cfg.whiten, random_state=cfg.random_state)
    pca.fit(X)
    return pca


def transform_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X)

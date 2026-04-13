# INF367A — Image2Biomass

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Task

Predict biomass dry weight (Dry_Green_g, Dry_Dead_g, Dry_Clover_g) from field images and tabular metadata. Two additional targets are derived: GDM_g (Green + Clover) and Dry_Total_g (Green + Dead + Clover). The competition metric is a weighted R² score where Dry_Total_g carries the largest weight (0.5).

## Pipeline Overview

1. **Data loading** — pivot the long-format CSV to one row per image, extract date features (year, month, day, dayofyear)
2. **Vision features** — extract 512-dimensional embeddings from a pretrained ResNet18 (penultimate layer), cached to `.npy` files
3. **PCA** — reduce vision features from 512 to 128 dimensions
4. **Merge & scale** — combine tabular metadata with PCA features, apply StandardScaler
5. **Regression** — fit a multi-output regressor to predict the three direct targets

## Novel Method: TabPFN

### What is TabPFN?

TabPFN (Tabular Prior-Fitted Network) is a foundation model for tabular data developed by Hollmann et al. (2025, TabPFN v2). Unlike traditional ML models that learn from scratch on each dataset, TabPFN is a transformer that was pre-trained on millions of synthetic tabular datasets sampled from a learned prior over structural causal models. At inference time, it treats the training data as context (similar to in-context learning in LLMs) and predicts test labels — no gradient-based fitting on the downstream task is required.

### How it works

TabPFN v2 uses a transformer architecture that was meta-learned via prior-fitting. During pre-training, the model learned to approximate Bayesian inference across diverse data distributions. Given a new dataset, TabPFN encodes the training samples as a sequence of tokens, appends the test inputs, and outputs predictions through attention — effectively performing implicit Bayesian model selection and hyperparameter tuning. TabPFN v2 improves on the original (v1) by supporting larger datasets through retrieval-augmented in-context learning, handling both classification and regression, and using an ensemble of multiple forward passes (controlled by the `n_estimators` parameter) for more robust predictions.

Key properties:
- **No hyperparameter tuning needed** — the model has already learned how to adapt to different data distributions during pre-training
- **Handles small-to-medium datasets well** — particularly strong when n < 10,000 samples, which matches our competition dataset
- **Ensemble-based** — aggregates predictions from multiple forward passes with different data orderings for robustness
- **Fast fitting** — no iterative training loop; fitting consists of preprocessing and storing the training data for the forward pass

### Why TabPFN for this task

Our dataset has ~700 training images with 128 PCA features + tabular metadata — a small-to-medium tabular regression problem, which is exactly TabPFN's strength. Traditional tree-based models like ExtraTreesRegressor require careful hyperparameter tuning and can overfit on small datasets. TabPFN's pre-trained prior acts as a strong regularizer, enabling it to generalize better without manual tuning.

### Implementation

TabPFN is integrated via the `ModelType` enum in the configuration. In [run.py](src/main/run.py) or [run_only_train.py](src/main/run_only_train.py), switching between models requires changing a single line:

```python
# Use TabPFN
train_cfg = TrainConfig(model_type=ModelType.TABPFN)

# Use ExtraTreesRegressor
train_cfg = TrainConfig(model_type=ModelType.EXTRA_TREES)
```

TabPFN is sklearn-compatible, so it plugs directly into the existing `MultiOutputRegressor` pipeline. The only adaptation is that `n_jobs` is set to 1 for TabPFN since it handles parallelism internally and cannot be pickled into worker processes.

## Model Evaluation

Both models were evaluated using 5-fold GroupKFold cross-validation (grouped by image_path to prevent data leakage), on a subset of 160 image groups.

### Weighted R² (competition metric)

| Model | Weighted R² |
|---|---|
| ExtraTreesRegressor | 0.640 |
| **TabPFN** | **0.747** |

### Per-target R²

| Target | ExtraTrees | TabPFN |
|---|---|---|
| Dry_Clover_g (w=0.1) | 0.284 | 0.548 |
| Dry_Dead_g (w=0.1) | 0.239 | 0.365 |
| Dry_Green_g (w=0.1) | 0.639 | 0.761 |
| GDM_g (w=0.2) | 0.524 | 0.683 |
| Dry_Total_g (w=0.5) | 0.501 | 0.644 |

TabPFN outperforms ExtraTreesRegressor on every target, with the largest improvements on Dry_Clover_g (+0.264) and Dry_Total_g (+0.143).

## Requirements

See `pyproject.toml`. The environment manager [uv](https://docs.astral.sh/uv/getting-started/installation/) is used:

```bash
pip install uv
uv sync
```

## How to run

```bash
cd src
uv run python -m main.run             # full pipeline: train + predict on test set
uv run python -m main.run_only_train   # training + CV evaluation only
```

## Formatting

We use [ruff](https://docs.astral.sh/ruff/) to format code.

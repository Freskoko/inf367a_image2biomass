# INF367A - Image2Biomass

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Task

Predict biomass dry weight (Dry_Green_g, Dry_Dead_g, Dry_Clover_g) from field images and some tabular metadata. Two extra targets are derived from these three: GDM_g = Green + Clover, and Dry_Total_g = Green + Dead + Clover. The competition score is a weighted R², where Dry_Total_g is weighted 0.5, GDM_g 0.2, and the rest 0.1.

## Pipeline

1. Load the long-format CSV and pivot it so each row is one image. Extract year/month/day/dayofyear from the sampling date.
2. Run a pretrained ResNet18 over the images and take the 512-d penultimate-layer features. Cache them to `.npy` so we don't redo this on every run.
3. Reduce the 512 ResNet features to 128 with PCA.
4. Merge the PCA features with the tabular metadata and one-hot encode categoricals (State, Species). Scale numerics with StandardScaler.
5. Fit a multi-output regressor on the three direct targets. The two composite targets are computed from the predictions afterwards.

## Novel Method: TabPFN

### What it is

TabPFN (Tabular Prior-Fitted Network) is a foundation model for tabular data from Hollmann et al. (2025, TabPFN v2). Instead of training from scratch on our dataset, TabPFN is a transformer that has already been pre-trained on millions of synthetic tabular datasets drawn from a prior over structural causal models. At inference time it takes the training set as in-context examples (similar to how LLMs use context) and outputs predictions for the test rows in a forward pass. There is no gradient-based training step on our data.

### How it works

During pre-training, TabPFN learned to approximate Bayesian inference across a wide range of data-generating processes sampled from its prior. When you call `.fit()` on a new dataset, the model mostly just stores and preprocesses the training data. When you call `.predict()`, the training set and the test inputs are fed through the transformer together, and the attention mechanism produces predictions that implicitly marginalize over plausible models. You can think of this as doing model selection and hyperparameter tuning "inside" the forward pass, using the knowledge baked in during pre-training.

Compared to the original v1 (classification-only, up to around 1,000 samples), TabPFN v2 adds regression support and scales to roughly 10,000 samples and 500 features thanks to a revised architecture and prior. It also ensembles multiple forward passes with different feature orderings and preprocessings (controlled by `n_estimators`) to stabilize predictions.

Main practical properties:

- No hyperparameter search needed. The prior already encodes what a good model looks like.
- Works well on small and medium tabular datasets, which matches what we have here.
- "Training" is essentially free; the cost is at prediction time, because each prediction is a transformer forward pass over the full training set.

### Why it fits this task

Our dataset is small (~700 images after pivoting, plus ~150 features once PCA + tabular + one-hot are combined). That's right in TabPFN's sweet spot. Tree ensembles like ExtraTrees work fine but need some hyperparameter tuning to avoid overfitting on small data. TabPFN's pre-trained prior acts as a strong regularizer without us having to tune anything.

### Implementation

TabPFN is wired in through a `ModelType` enum. Both entry-point scripts take a `--model` argument, so switching between models is just a CLI flag, no code edits. Internally, `TrainConfig(model_type=...)` decides what `get_model()` returns, and the rest of the pipeline (preprocessor, `MultiOutputRegressor`, `Pipeline`) is the same for both models.

Two small adjustments for TabPFN:

- `MultiOutputRegressor` runs with `n_jobs=1` when using TabPFN. TabPFN can't be pickled into spawned worker processes and does its own parallelism anyway.
- `random_state` is forwarded to `TabPFNRegressor` so the ensembled forward passes are reproducible.

## Evaluation

Both models were evaluated with 5-fold GroupKFold cross-validation, grouped by `image_path` so no image ends up in both train and validation of a fold.

Setup for the numbers below (the default pipeline):

- Vision backbone: pretrained ResNet18, 512-d features
- PCA to 128 components on the ResNet features
- Tabular features: `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, plus date features
- StandardScaler on numerics, one-hot on categoricals
- 5 folds, 160 image groups, seed 42
- Metric: weighted R² (same weights as the competition)

If the vision backbone is swapped (e.g. DINO / DINOv2) or the PCA dimensionality changes, the absolute numbers will move, so rerun both models before comparing.

### Weighted R² (competition metric)

| Model               | Weighted R² |
|---------------------|-------------|
| ExtraTreesRegressor | 0.640       |
| **TabPFN**          | **0.747**   |

### Per-target R²

| Target                | ExtraTrees | TabPFN |
|-----------------------|------------|--------|
| Dry_Clover_g (w=0.1)  | 0.284      | 0.548  |
| Dry_Dead_g   (w=0.1)  | 0.239      | 0.365  |
| Dry_Green_g  (w=0.1)  | 0.639      | 0.761  |
| GDM_g        (w=0.2)  | 0.524      | 0.683  |
| Dry_Total_g  (w=0.5)  | 0.501      | 0.644  |

TabPFN is better on every target. The biggest jumps are Dry_Clover_g (+0.26) and Dry_Total_g (+0.14), which matters because Dry_Total_g carries the largest weight in the final score.

### Reproducing

```bash
cd src
uv run python -m main.run_only_train --model extra_trees
uv run python -m main.run_only_train --model tabpfn
```

## Requirements

See `pyproject.toml`. We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the environment:

```bash
pip install uv
uv sync
```

## How to run

```bash
cd src

# Full pipeline: train + predict on the test set, writes submission.csv
uv run python -m main.run --model tabpfn
uv run python -m main.run --model extra_trees

# Training + 5-fold CV evaluation only (no test predictions)
uv run python -m main.run_only_train --model tabpfn
uv run python -m main.run_only_train --model extra_trees
```

If you don't pass `--model`, TabPFN is used by default.

## Formatting

We use [ruff](https://docs.astral.sh/ruff/) to format the code.

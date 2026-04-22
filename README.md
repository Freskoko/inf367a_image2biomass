# INF367A - Image2Biomass

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Task

Using a dataset for precision agriculture, the task is to predict pasture biomass from top-view images and some tabular metadata. Pasture biomass is defined as dry weight including:

• Dry_Green_g: Green vegetation other than clover (grams)

• Dry_Dead_g: Senescent material (grams)

• Dry_Clover_g: Clover component (grams)

• GDM_g: Green dry matter, calculated as the sum of green vegetation and clover (grams). GDM_g = Green + Clover

• Dry_Total_g: Total biomass, combining all components (grams). Dry_Total_g = Green + Dead + Clover

The competition score is a weighted R², where Dry_Total_g is weighted 0.5, GDM_g weighted 0.2, and the rest 0.1.


## Pipeline

1. Load the long-format CSV and pivot it so each row is one image. The tabular metadata (`State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`, `Sampling_Date`) isn't used as features — `test.csv` doesn't contain these columns, so the production pipeline is vision-only and CV is kept symmetric with it.
2. Run a pretrained vision backbone (DINOv2 by default, ResNet18 or ConvNeXt-Tiny via `--vision-backbone resnet|convnext`) over the images and take the penultimate-layer features. Cache them to `.npy` so we don't redo this on every run.
3. Inside the sklearn `Pipeline`, fit `StandardScaler → PCA(n_components=128)` on the training split. Under CV both refit per fold so validation-fold statistics don't leak into the PCA basis or the scaler.
4. Fit a multi-output regressor on the three direct targets (`Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`). The two composite targets (`GDM_g`, `Dry_Total_g`) are computed from those predictions afterwards.

## Regressors

Two regressors are available: the baseline `ExtraTreesRegressor` and the novel method `TabPFN`.

**TabPFN** (Tabular Prior-Fitted Network; Hollmann et al., 2025, v2) is a transformer pretrained on millions of synthetic tabular datasets. It performs prediction in a single forward pass over the training set used as in-context examples — no per-task training and no hyperparameter search. It fits our setting because the dataset is small (~357 images after pivot, 128 PCA-reduced vision features), well within TabPFN's ~10k-sample / 500-feature range. See [docs/TabPFN.md](docs/TabPFN.md) for full implementation notes and evaluation results.

## Requirements

See `pyproject.toml`. We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the environment:

```bash
pip install uv
uv sync
```

If using a virtual environment, run this:
uv venv .venv ### Create a venv
uv sync ### Install exactly what’s in uv.lock in the venv
. venv\Scripts\activate ### Activate the venv

## How to run

Both scripts take two flags:

- `--model`: `tabpfn` (default) or `extra_trees`
- `--vision-backbone`: `dino` (default), `resnet`, or `convnext` (`torchvision.models.convnext_tiny`)

```bash
cd src

# Full pipeline: train on the training set, predict on the test set, write submission.csv
uv run python -m main.run                                        # defaults (tabpfn + dino)
uv run python -m main.run --model extra_trees
uv run python -m main.run --model tabpfn      --vision-backbone resnet
uv run python -m main.run --model extra_trees --vision-backbone resnet

# Training + 5-fold CV evaluation only (no test predictions)
uv run python -m main.run_only_train                             # defaults (tabpfn + dino)
uv run python -m main.run_only_train --model extra_trees
uv run python -m main.run_only_train --model tabpfn      --vision-backbone resnet
uv run python -m main.run_only_train --model extra_trees --vision-backbone resnet
```

## Formatting

We use [ruff](https://docs.astral.sh/ruff/) to format the code.


## Lower Resources for faster training
Change this parameter in utils to debug and speed up the train process

train_cfg.lower_resources = True

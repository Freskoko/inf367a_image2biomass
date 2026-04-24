# INF367A - Image2Biomass

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Files included in submission

1. This file
2. A zip folder including the repo for our CcGAN project (inf367a_image2biomass.zip)
3. A PDF report
4. A data exploration notebook, which can be found here: `src/exploration/explore.ipynb`


## Individual work vs group work

In `/docs` at the root of this repo there are readmes describing the individual work.

| Person | Project | Writeup | Corresponding code |
| :--- | :--- | :--- | :--- |
| Henrik Brøgger | CcGAN | `docs/CcGAN.md` | `src/main/ccgan_improved/` |
| Kristofers Gulbis | TabPFN | `docs/TabPFN.md` | `src/main/utils/utils.py`, `src/main/regression/baseline_training.py`, `src/main/run.py`, `src/main/run_only_train.py` |

The PDF report that we submit with the code is group work.

## Task

Using a dataset for precision agriculture, the task is to predict pasture biomass from top view images and some tabular metadata. Pasture biomass is defined as dry weight including:

• Dry_Green_g: Green vegetation other than clover (grams)

• Dry_Dead_g: Senescent material (grams)

• Dry_Clover_g: Clover component (grams)

• GDM_g: Green dry matter, calculated as the sum of green vegetation and clover (grams). GDM_g = Green + Clover

• Dry_Total_g: Total biomass, combining all components (grams). Dry_Total_g = Green + Dead + Clover

The competition score is a weighted R², where Dry_Total_g is weighted 0.5, GDM_g weighted 0.2, and the rest 0.1.


## Pipeline

1. Load the long format CSV and pivot it so each row is one image. We don't use the tabular metadata (`State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`, `Sampling_Date`) as features, because `test.csv` doesn't have those columns. Keeping the pipeline vision only means CV matches what actually runs at test time.
2. Run a pretrained vision backbone (DINOv2 by default, ResNet18 or ConvNeXt Tiny via `--vision-backbone resnet|convnext`) over the images and take the penultimate layer features. We cache them to `.npy` so we don't have to redo this every run.
3. Inside the sklearn `Pipeline`, fit `StandardScaler` and then `PCA(n_components=128)` on the training split. Under CV both refit per fold, so no validation fold statistics leak into the PCA basis or the scaler.
4. Fit a multi output regressor on the three direct targets (`Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`). The two composite targets (`GDM_g`, `Dry_Total_g`) are computed from those predictions afterwards.

## Regressors

We have two regressors to choose from: `ExtraTreesRegressor` as the baseline and `TabPFN` as the new method.

TabPFN (Tabular Prior Fitted Network, Hollmann et al. 2025, v2) is a transformer that was pretrained once on millions of synthetic tabular datasets. At inference it takes the training set as in context examples and predicts the test rows in one forward pass, so there is no per task training and no hyperparameter tuning to do. Our dataset is small (around 357 images after pivoting, 128 PCA features), which is well within what TabPFN is designed for. See [docs/TabPFN.md](docs/TabPFN.md) for the full description, implementation notes and results.

Under 5 fold CV on a 160 image group subsample, TabPFN with ConvNeXt Tiny features was the best combination (0.662 global weighted R2, 0.544 per target weighted R2) and TabPFN beat ExtraTrees on every (target, backbone) combination. On the Kaggle leaderboard the ranking flips: ExtraTrees with DINO tops the private leaderboard at 0.260 and TabPFN's best private score is 0.232 (ConvNeXt). See [docs/TabPFN.md](docs/TabPFN.md) for the full discussion of the CV to Kaggle gap.

## Requirements

See `pyproject.toml`. We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the environment:

```bash
pip install uv
uv sync
```

If using a virtual environment:

```bash
uv venv .venv                       # Create a venv
uv sync                             # Install exactly what's in uv.lock
source .venv/bin/activate           # Activate the venv (macOS/Linux)
# .venv\Scripts\activate            # Activate the venv (Windows)
```

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
uv run python -m main.run --model tabpfn      --vision-backbone convnext
uv run python -m main.run --model extra_trees --vision-backbone convnext

# Training + 5-fold CV evaluation only (no test predictions)
uv run python -m main.run_only_train                             # defaults (tabpfn + dino)
uv run python -m main.run_only_train --model extra_trees
uv run python -m main.run_only_train --model tabpfn      --vision-backbone resnet
uv run python -m main.run_only_train --model extra_trees --vision-backbone resnet
uv run python -m main.run_only_train --model tabpfn      --vision-backbone convnext
uv run python -m main.run_only_train --model extra_trees --vision-backbone convnext
```

## Formatting

We use [ruff](https://docs.astral.sh/ruff/) to format the code.


## Faster CV runs (default)

By default `TrainConfig.lower_resources = True`, which subsamples CV to 160 image groups so TabPFN runs finish in a couple of minutes on CPU. If you want a full CV on all images, set it to `False` in [src/main/utils/utils.py](src/main/utils/utils.py) (the `TrainConfig` dataclass) or construct `TrainConfig(lower_resources=False)` from the call site.

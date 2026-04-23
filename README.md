# INF367A - Image2Biomass

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Files included in submission

1. This file
2. A zip folder including the repo for our CcGAN project (inf367a_image2biomass.zip)
3. A PDF report
4. A data exploration notebook, which can be found here: `src/exploration/explore.ipynb`


## Individual work vs group work

In /docs at the base of this repo, there are readmes describing each of the induvidual work done.

docs/CcGAN.md

| Person | Project  | Folder  | Corresponding code
| :--- | :--- | :--- | :--- |
| **Henrik Brøgger** | CcGAN | docs/CcGAN.md | src/main/ccgan_improved |
| **Kristofers Gulbis** | TabPFN | docs/TabPFN.md | |
# TODO: WRITE YOUR NAMES HERE!!!

The pdf report that we submit with the code is group work.

## Task

Using a dataset for precision agriculture, the task is to predict pasture biomass from top-view images and some tabular metadata. Pasture biomass is defined as dry weight including:

• Dry_Green_g: Green vegetation other than clover (grams)

• Dry_Dead_g: Senescent material (grams)

• Dry_Clover_g: Clover component (grams)

• GDM_g: Green dry matter, calculated as the sum of green vegetation and clover (grams). GDM_g = Green + Clover

• Dry_Total_g: Total biomass, combining all components (grams). Dry_Total_g = Green + Dead + Clover

The competition score is a weighted R², where Dry_Total_g is weighted 0.5, GDM_g weighted 0.2, and the rest 0.1.


## Pipeline

1. Load the long-format CSV and pivot it so each row is one image. Extract year/month/day/dayofyear from the sampling date.
2. Run a pretrained vision backbone (DINOv2 by default, ResNet18 or ConvNeXt-Tiny via `--vision-backbone resnet|convnext`) over the images and take the penultimate-layer features. Cache them to `.npy` so we don't redo this on every run.
3. Reduce the vision features to 128 components with PCA.
4. Merge the PCA features with the tabular metadata and one-hot encode categoricals (State, Species). Scale numerics with StandardScaler.
5. Fit a multi-output regressor on the three direct targets. The two composite targets are computed from the predictions afterwards.

Two regressors are available: the baseline `ExtraTreesRegressor` and the novel method `TabPFN`. See [docs/TabPFN.md](docs/TabPFN.md) for the TabPFN description, implementation notes, and evaluation results.

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

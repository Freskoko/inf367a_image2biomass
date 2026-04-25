# INF367A - Image2Biomass

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Files included in submission

1. This file
2. A zip of the full project repo (`inf367a_image2biomass.zip`), which contains both the TabPFN and CcGAN work
3. A PDF report
4. A data exploration notebook at `src/exploration/explore.ipynb` (inside the zip)


## Individual work vs group work

In `/docs` at the root of this repo there are readmes describing the individual work.

| Person | Project | Writeup | Corresponding code |
| :--- | :--- | :--- | :--- |
| Henrik Brøgger | CcGAN | `docs/CcGAN.md` | `src/main/ccgan_improved/` |
| Kristofers Gulbis | TabPFN | `docs/TabPFN.md` | `src/main/utils/utils.py`, `src/main/regression/baseline_training.py`, `src/main/run.py`, `src/main/run_only_train.py` |

The PDF report that we submit with the code is group work.

## Repository structure

```
docs/
  TabPFN.md            Writeup for the TabPFN novel method (Kristofers)
  CcGAN.md             Writeup for the CcGAN novel method (Henrik)
  images/              Figures used in the writeups
src/
  data/                train.csv, test.csv, sample_submission.csv, train/ and test/ image folders
  exploration/
    explore.ipynb      Data exploration notebook (target distributions, image checks, sanity plots)
  main/
    run.py             Full pipeline entry point: trains and writes submission.csv
    run_only_train.py  CV-only entry point: 5 fold GroupKFold evaluation
    utils/             TrainConfig, VisionModelConfig, DatasetPaths, ModelType enum, TABPFN_TOKEN
    wrangling/         CSV loading, long to wide pivot, vision feature extraction driver
    preprocessing/     Pivot helpers, date features, composite target sanity checks
    vision/            DINOv2, ResNet18 and ConvNeXt Tiny feature extractors
    regression/        Pipeline builder, CV loop, weighted R2 metrics, fit/predict
    ccgan_improved/    Henrik's CcGAN code (separate from the main pipeline)
pyproject.toml         Dependencies and uv config
README.md              This file
```

## Data exploration

Before any modelling we ran some basic exploration in [src/exploration/explore.ipynb](src/exploration/explore.ipynb): target distributions, sampling date coverage, per State and per Species splits, a few sample images per target range, and sanity checks that `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g` actually holds in the data (one row off by 0.31 g, the rest fine). This is what informed a few choices like keeping the pipeline vision only (since `test.csv` doesn't have the tabular columns) and recomputing composite targets from base predictions.

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

## Novel methods:


Our baseline is ExtraTrees on top of pretrained ResNet18 features. It's a tree ensemble on standard ImageNet representations, deliberately conventional so we have a sensible reference point to measure improvements against. From there each group member contributed one novel method, and each method targets a specific weakness of the baseline.

### DINOv2 (Lang), replaces the vision backbone

ImageNet-pretrained ResNet18 was trained to discriminate between object categories like cats, cars, and chairs. Our task is the opposite. Every image contains the same kind of object (grass), and the relevant signal is fine-grained texture composition: how much of the image is green vegetation, how much is dead matter, how much is clover. Classification-trained CNNs are known to discard texture detail in favour of shape and object identity, which is the wrong inductive bias here. DINOv2 [Oquab et al., 2023] is trained without labels using self-distillation on a curated 142M-image dataset, and it preserves fine-grained spatial and textural structure that supervised pretraining tends to throw away. It also has roughly two orders of magnitude more pretraining data than the original ResNet18 ImageNet weights, so we expected it to transfer better to a domain (top-down pasture imagery) that is quite far from ImageNet.

### ConvNeXt-Tiny (Ole), replaces the vision backbone
ConvNeXt [Liu et al., 2022] is a 2022 redesign of convolutional architectures that incorporates several ideas from vision transformers: a patchify stem, depthwise separable convolutions, larger kernel sizes, inverted bottlenecks, and LayerNorm in place of BatchNorm. The combination closes most of the gap to ViTs on ImageNet while keeping convolutional inductive biases (translation equivariance, locality), which we expected to matter on a small dataset where transformers can struggle. We chose it as a counterpoint to DINOv2. If a stronger supervised CNN matches or beats DINOv2 here, it suggests the data scaling matters more than the self-supervision objective. If DINOv2 wins, it suggests the opposite. Either way the comparison tells us something.

### TabPFN (Kristofers), replaces the regression head
The baseline tree ensemble has a known weakness on small data. It benefits wfrom per-dataset hyperparameter tuning (n_estimators, depth, leaf size), but cross-validation-based tuning is itself noisy when the dataset is small, so the search becomes a source of overfitting on its own. TabPFN [Hollmann et al., 2025] avoids this entirely. It is a transformer pretrained once on millions of synthetic tabular datasets sampled from a prior over structural causal models, so its forward pass approximates a posterior predictive distribution under that prior. Model selection and hyperparameter marginalisation happen inside the network rather than before it. The operating regime it was designed for (up to 10k samples, 500 features) covers our task (~357 images, 128 PCA features) with room to spare. The no-tuning property is exactly what we want here, since the CV signal we would otherwise tune against is itself noisy at this sample size.

### CcGAN (Henrik), expands the dataset
The first three methods all assume a fixed dataset and try to extract more signal from it. CcGAN [Ding et al., 2021] takes the orthogonal approach. It generates new training images conditioned on a continuous biomass target, which in principle could fill in underrepresented regions of the label distribution. Standard class-conditional GANs only handle discrete labels. CcGAN's contribution is a vicinal discriminator loss that lets the model condition on a continuous scalar (for us, total biomass in grams). If it had worked, it would have addressed data scarcity at the source rather than working around it downstream. It did not work well at the resolution our compute budget allowed (see docs/CcGAN.md), which is itself a useful negative result.

### How they fit together

The three pipeline-replacement methods (DINOv2, ConvNeXt-Tiny, TabPFN) plug into a modular sklearn pipeline through the --vision-backbone and --model CLI flags, so every combination of backbone and regressor can be evaluated under the same CV protocol. That means any improvement we report is attributable to the specific component we swapped, not to incidental pipeline differences. The CcGAN branch is self-contained because it operates on the data itself rather than on the pipeline.

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


## Extra notes

Some of us used AI to help with coding, debbuging code, structuring readmes and some other issues.
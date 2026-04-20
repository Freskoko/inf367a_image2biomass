# Novel Method: TabPFN

## What it is

TabPFN (Tabular Prior-Fitted Network) is a foundation model for tabular data from Hollmann et al. (2025, TabPFN v2). My in-class presentation covered TabPFN v1; for this implementation I used v2 (course staff confirmed this is allowed), which is the version that supports regression. Instead of training from scratch on our dataset, TabPFN is a transformer that has already been pretrained on millions of synthetic tabular datasets generated from a prior over structural causal models. At inference it takes the training set as in context examples (similar to how LLMs use context) and outputs predictions for the test rows in a forward pass.

## How it works

During pre training, TabPFN learned to approximate Bayesian inference across a wide range of data generating processes sampled from its prior. When you call `.fit()` on a new dataset, the model mostly just stores and preprocesses the training data. When you call `.predict()`, the training set and the test inputs are fed through the transformer together, and attention produces predictions that implicitly marginalize over plausible models. You can kind of think of it as doing model selection and hyperparameter tuning "inside" the forward pass, using the knowledge the authors gave it during pre training.

Compared to the original v1 (classification only, up to around 1,000 samples), TabPFN v2 adds regression support and scales to roughly 10,000 samples and 500 features because of the improved architecture. It also ensembles multiple forward passes with different feature orderings and preprocessings (controlled by `n_estimators`) to stabilize predictions.

Main practical properties:

- No hyperparameter search needed. The prior already encodes what a good model looks like.
- Works well on small and medium tabular datasets, which matches what we have in this exercise.
- "Training" is basically free, the cost is at prediction time, because every prediction is a transformer forward pass over the full training set.

## Why does it fit this task

Our dataset is small (~700 images, plus ~150 features). That's in the range TabPFN was designed for. Tree ensembles like ExtraTrees work fine but need some hyperparameter tuning to not overfit on small data.

## Implementation

Getting TabPFN running took more work than I thought it would. Here are the main things I ran into:

**Getting the model to download.** TabPFN doesn't ship the weights inside the pip package. The first time you call `.fit()` it tries to download them from Prior Labs, and the download is behind a license you have to get. So first i got `TabPFNLicenseError` until i registered on `ux.priorlabs.ai`, accepted everything, copied the API key from your account page, and set it as `TABPFN_TOKEN`. I dropped the token into `utils.py` and set it as an environment variable when the module is imported, so anyone cloning the repo can just run the scripts without doing the browser dance themselves. The token is read only, so it's fine to keep in the repo, no need to hide it as an env var.

**Making it work with multiple targets.** TabPFN only predicts one target at a time, but we have three (Dry_Green_g, Dry_Dead_g, Dry_Clover_g). The code already uses `MultiOutputRegressor` for exactly this situation, so I figured TabPFN would just slot in. That didn't work, and figuring out why took a while. The first failure looked like a license error even though I'd set the token correctly, which was confusing. Reading the errors more carefully it turned out the error was actually raised inside a worker process, not in my main one. With `n_jobs=-1`, `MultiOutputRegressor` trains the three copies (one per target) in parallel worker processes. On macOS those workers start with the `spawn` method, which doesn't get the parent's `os.environ` changes, so `TABPFN_TOKEN` was fine in my main process but missing in the workers. On top of that, TabPFN objects don't pickle cleanly into workers anyway (they hold state that doesn't survive serialization). I tried a couple of things first (forcing the env var earlier in import order, constructing the model a bit differently) before settling on the simpler fix: detect the TabPFN case in `model_wrapper_creator` and force `n_jobs=1` for it so everything runs in one process. TabPFN does its own parallelism internally so the time increase from dropping external parallelism is pretty small.

**Reproducibility.** TabPFN runs several forward passes with shuffled features and averages them. If you don't set `random_state` it picks its own, which means two CV runs give slightly different numbers. Not a bug, just annoying when you're trying to compare models. I pass `TrainConfig.random_state` into the constructor so runs are repeatable.

**Switching between models.** I wanted it to be easy for the rest of the group to try different setups (different vision backbones etc.) without having to edit any code to change the regressor. So I added a `ModelType` enum and a `--model` flag on both `run.py` and `run_only_train.py`. Picking a model is just a CLI argument, no code edits.

**Making CV fast enough.** TabPFN's prediction time scales with the size of the training set, because every prediction is a transformer forward pass over all the training rows. Running a full 5 fold CV over the whole dataset was too slow to iterate on, so I used the `lower_resources` flag that was already in the code. It samples 160 image groups and runs CV on those. That's enough to compare models, and a full CV with TabPFN takes about 2 minutes on CPU instead of way longer.

## How the data reaches TabPFN

Once all of that is in place, I never call TabPFN's API directly. Everything goes through the sklearn `Pipeline` that `model_wrapper_creator` builds, so the same code path handles both ExtraTrees and TabPFN.

When `pipe.fit(X, y)` is called (either in `fit_full` for the full run, or inside each CV fold in `cv_mean_r2`):

1. The `ColumnTransformer` one-hot encodes `State` / `Species` and passes the numerics (PCA features, NDVI, height, date features) through unchanged.
2. `MultiOutputRegressor` clones the `TabPFNRegressor` once per target column (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`) and calls `.fit(X_processed, y_col)` on each clone. TabPFN's "fit" mostly just preprocesses and stores the training rows as context.
3. At `pipe.predict(X_test)`, each clone runs a forward pass over its training context plus the test rows and returns predictions for its target. `MultiOutputRegressor` stacks the three columns back together. The two composite targets (`GDM_g`, `Dry_Total_g`) are then computed from those three predictions.

The rest of the pipeline (loading data, vision features, PCA, scaling, CV, metrics) is shared with the ExtraTrees path.

## Files I changed (individual work)

None of the changes is huge in line count, but a lot of them came out of debugging rather than just writing code, so I want to be clear about what was done.

**`pyproject.toml`** — added `tabpfn>=2.0.0` to `dependencies`. The v2 pin matters because v1 is classification only, v2 is the one with regression. `uv sync` pulls in tabpfn and its transitive stuff (torch, huggingface-hub, and a couple of Prior Labs telemetry/auth packages).

**`src/main/utils/utils.py`** — this is where most of the TabPFN related code ended up:

- Added a `ModelType` enum (`EXTRA_TREES`, `TABPFN`) with a `from_string` classmethod so the `--model` CLI string maps to a typed value and you get a clear error on typos.
- Added `model_type: ModelType = ModelType.EXTRA_TREES` to `TrainConfig` so the choice flows through the rest of the code via the config object.
- Rewrote `get_model()` to dispatch on `self.model_type`. When TabPFN is selected import `TabPFNRegressor` inside the function so `--model extra_trees` runs don't pay the tabpfn/torch import cost, and so the project doesn't break for anyone who hasn't run `uv sync` with the full dependency set yet. I also pass `random_state=self.random_state` in explicitly, otherwise TabPFN uses its own default and CV becomes non deterministic (see the reproducibility part above).
- Added the `TABPFN_TOKEN` constant and set `os.environ["TABPFN_TOKEN"] = ...` at module import time. Setting it inside `get_model()` would be too late, because sklearn can spawn workers before we ever call `get_model`, and those workers start with whatever environment exists at spawn time. Doing it at module import means the token is in the environment before any user code runs.

**`src/main/regression/baseline_training.py`** — `model_wrapper_creator` now picks `n_jobs=1` when `train_cfg.model_type == ModelType.TABPFN`, and otherwise uses `train_cfg.n_jobs` like before (so ExtraTrees behavior is unchanged). This is the one-line fix for the worker-process issue described above, but it was the thing that took me the longest to actually find.

**`src/main/run.py` and `src/main/run_only_train.py`** — both entry-point scripts now take a `--model` argument via `argparse`, with choices `tabpfn` / `extra_trees` and default `tabpfn`. The string goes through `ModelType.from_string(args.model)` into `TrainConfig(model_type=...)`. Everything else in those scripts is unchanged. So picking a model is really just a CLI flag, no edits.

**On purpose left alone:** the preprocessing (`_build_preprocessor`, `ColumnTransformer`, `MultiOutputRegressor`, the `Pipeline` wrapping), the CV loop (`cv_mean_r2`) and the metric code (`weighted_r2_global`). Keeping all of that shared is what makes the ExtraTrees vs TabPFN numbers actually comparable, the only thing that differs between the two runs is the estimator inside `MultiOutputRegressor`.

## Evaluation

Both models were evaluated with 5 fold GroupKFold cross validation, grouped by `image_path` so no image ends up in both train and validation of a fold.

Setup:

- Vision backbone: either pretrained ResNet18 (512-d features) or DINOv2 (`dinov2_vits14`)
- PCA to 128 components on the vision features
- Tabular features: `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, plus date features
- StandardScaler on numerics, one-hot on categoricals
- 5 folds, 160 image groups, seed 42
- Metric: weighted R² (same weights as the competition: 0.5 on Dry_Total_g, 0.2 on GDM_g, 0.1 on the rest)

To keep the comparison fair, TabPFN and ExtraTrees were run on the exact same preprocessed features. Swapping the vision backbone (ResNet vs DINO) moves the absolute numbers around but lets me check whether TabPFN's advantage holds across different feature types.

### Weighted R² (competition metric)

| Model                             | Weighted R² |
|-----------------------------------|-------------|
| ExtraTreesRegressor with ResNet   | 0.640       |
| TabPFN with ResNet                | 0.747       |
| ExtraTreesRegressor with DINO     | 0.763       |
| **TabPFN with DINO**              | **0.864**   |

### Per-target R²

| Target                | ExtraTrees_ResNet | TabPFN_ResNet | ExtraTrees_DINO | TabPFN_DINO |
|-----------------------|-------------------|---------------|-----------------|-------------|
| Dry_Clover_g (w=0.1)  | 0.284             | 0.548         | 0.646           | 0.724       |
| Dry_Dead_g   (w=0.1)  | 0.239             | 0.365         | 0.424           | 0.588       |
| Dry_Green_g  (w=0.1)  | 0.639             | 0.761         | 0.746           | 0.862       |
| GDM_g        (w=0.2)  | 0.524             | 0.683         | 0.721           | 0.843       |
| Dry_Total_g  (w=0.5)  | 0.501             | 0.644         | 0.682           | 0.819       |

TabPFN beats ExtraTrees on every target with both backbones. The gap is bigger with DINO features, but it's there with ResNet too, so the improvement isn't just a feature extractor thing.

### Reproducing

```bash
cd src
# ResNet features
uv run python -m main.run_only_train --model extra_trees --vision-backbone resnet
uv run python -m main.run_only_train --model tabpfn      --vision-backbone resnet
# DINO features
uv run python -m main.run_only_train --model extra_trees --vision-backbone dino
uv run python -m main.run_only_train --model tabpfn      --vision-backbone dino
```

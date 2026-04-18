# Novel Method: TabPFN

### What it is

TabPFN (Tabular Prior-Fitted Network) is a foundation model for tabular data from Hollmann et al. (2025, TabPFN v2). Instead of training from scratch on our dataset, TabPFN is a transformer that has already been pre trained on millions of synthetic tabular datasets drawn from a prior over structural causal models. At inference time it takes the training set as in-context examples (similar to how LLMs use context) and outputs predictions for the test rows in a forward pass. There is no gradient-based training step on our data.

### How it works

During pre training, TabPFN learned to approximate Bayesian inference across a wide range of data-generating processes sampled from its prior. When you call `.fit()` on a new dataset, the model mostly just stores and preprocesses the training data. When you call `.predict()`, the training set and the test inputs are fed through the transformer together, and the attention mechanism produces predictions that implicitly marginalize over plausible models. You can think of this as doing model selection and hyperparameter tuning "inside" the forward pass, using the knowledge baked in during pre-training.

Compared to the original v1 (classification-only, up to around 1,000 samples), TabPFN v2 adds regression support and scales to roughly 10,000 samples and 500 features thanks to a revised architecture and prior. It also ensembles multiple forward passes with different feature orderings and preprocessings (controlled by `n_estimators`) to stabilize predictions.

Main practical properties:

- No hyperparameter search needed. The prior already encodes what a good model looks like.
- Works well on small and medium tabular datasets, which matches what we have here.
- "Training" is essentially free; the cost is at prediction time, because each prediction is a transformer forward pass over the full training set.

### Why does it fit this task

Our dataset is small (~700 images after pivoting, plus ~150 features once PCA + tabular + one-hot are combined). That's in the range TabPFN was designed for. Tree ensembles like ExtraTrees work fine but need some hyperparameter tuning to avoid overfitting on small data. Since we didn't do any tuning for ExtraTrees either, we thought it would be a fair comparison to bring in a model whose main selling point is that it doesn't need tuning.

### Implementation

Getting TabPFN running took more work than i thought it would. Here are the main things i had to deal with:

**Getting the model to download.** TabPFN doesn't ship the weights inside the pip package. The first time you call `.fit()` it tries to download them from Prior Labs, and that download is locked behind a license you have to accept in a browser. So on a fresh machine you get a `TabPFNLicenseError` until you register on `ux.priorlabs.ai`, click accept, copy the API key from your account page, and set it as `TABPFN_TOKEN`. I dropped the token into `utils.py` and set it as an environment variable when the module is imported, so anyone cloning the repo can just run the scripts without doing the browser dance themselves. The token is read only (inference only), so having it in the repo is fine.

**Making it work with multiple targets.** TabPFN only predicts one target at a time, but we have three (Dry_Green_g, Dry_Dead_g, Dry_Clover_g). The code already uses `MultiOutputRegressor` for this, so we figured TabPFN would just slot in. It didn't. With `n_jobs=-1`, `MultiOutputRegressor` trains the three copies in parallel worker processes, and TabPFN can't be sent into those workers. On top of that, the `TABPFN_TOKEN` env var we set in the main process doesn't get passed to spawned workers on macOS, so the workers hit the license check and crash. I ended up setting `n_jobs=1` specifically for TabPFN. Everything runs in one process, and TabPFN does its own parallelism internally anyway, so we don't really lose anything.

**Reproducibility.** TabPFN runs several forward passes with shuffled features and averages them. If you don't set `random_state` it picks its own, which means two CV runs give different numbers. Not a bug, but annoying when you're trying to compare things. I pass `TrainConfig.random_state` into the constructor so runs are repeatable.

**Switching between models.** I wanted it to be easy for the rest of the group to try different setups (different vision backbones, etc.) without changing which regressor they use. So i added a `ModelType` enum and a `--model` flag on both `run.py` and `run_only_train.py`. Picking a model is just a CLI argument, no code changes needed.

**Making CV fast enough.** TabPFN's prediction time grows with the size of the training set, because every prediction is a transformer forward pass over all the training rows. Doing a full 5-fold CV over the whole dataset was too slow to iterate on, so we used the `lower_resources` flag that was already in the code. It samples 160 image groups and runs CV on those. That's enough to compare models, and a full CV with TabPFN takes about 2 minutes on CPU instead of much longer.

### How the data reaches TabPFN

Once the above pieces are in place, i never call TabPFN's API directly. Everything goes through the sklearn `Pipeline` that `model_wrapper_creator` builds, so the same code path serves both ExtraTrees and TabPFN.

When `pipe.fit(X, y)` is called (in `fit_full` for the full run, or inside each CV fold in `cv_mean_r2`):

1. The `ColumnTransformer` one-hot encodes `State` / `Species` and passes the numerics (PCA features, NDVI, height, date features) through unchanged.
2. `MultiOutputRegressor` clones the `TabPFNRegressor` once per target column (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`) and calls `.fit(X_processed, y_col)` on each clone. TabPFN's "fit" mostly preprocesses and stores the training rows as context for the transformer.
3. At `pipe.predict(X_test)`, each clone runs a forward pass over its training context plus the test rows and returns predictions for its target. `MultiOutputRegressor` stacks the three columns back together. The two composite targets (`GDM_g`, `Dry_Total_g`) are then computed from those three predictions.

The TabPFN-specific surface area in our code lives in three spots:

- `get_model()` in `utils.py` returns `TabPFNRegressor(random_state=...)` when `ModelType.TABPFN` is selected.
- `model_wrapper_creator` in `baseline_training.py` picks `n_jobs=1` for TabPFN and wraps the estimator in `MultiOutputRegressor` + `Pipeline`.
- `TABPFN_TOKEN` is set at the top of `utils.py` so the download + license check works on a fresh clone.

The rest of the pipeline (loading data, vision features, PCA, scaling, CV, metrics) is shared with the ExtraTrees path.
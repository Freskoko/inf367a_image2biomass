# Novel Method: ConvNeXt (Vision Backbone)

## What it is

ConvNeXt is a pure convolutional neural network architecture designed to close the gap to Vision Transformers. It was introduced by Zhuang Liu et al. (2022) in "A ConvNet for the 2020s". The core idea is simple: take a ResNet and replace its outdated design choices with ones that proved effective in transformer models. This includes larger kernels, depthwise convolutions, fewer nonlinearities, and different normalization. The result is still a CNN, but one that performs on par with transformer-based models like Swin Transformer while staying simpler to run. This project uses `convnext_tiny` from torchvision with ImageNet pretrained weights, strictly as a frozen feature extractor.

## How it works

### Starting point: a modernised ResNet

ConvNeXt is not designed from scratch. It is a step-by-step modification of a ResNet-50. Each change aligns the architecture more closely with what works in transformers.

The key changes:

**Patchify stem.** The initial layer becomes a 4×4 convolution with stride 4. This aggressively reduces resolution immediately, similar to how ViTs split images into patches.

**Depthwise convolutions with larger kernels.** Spatial mixing is done with depthwise convolutions. Each channel is processed independently, then mixed using 1×1 convolutions. The 7×7 kernel increases receptive field without increasing parameters significantly.

**Inverted bottleneck.** Channels expand inside the block instead of shrinking. Computation happens in a high-dimensional space, similar to transformer MLP layers.

**Fewer activation functions and normalisation layers.** Most ReLUs are removed. BatchNorm is replaced with LayerNorm. Each block contains fewer nonlinearities and one normalization, mirroring transformer structure.

**Separate downsampling layers.** Resolution reduction is moved outside residual blocks into dedicated layers. This mirrors patch merging in transformer models.

Individually, these changes are known. Combined, they produce a CNN that behaves like a transformer without using attention.

### The Tiny variant

The ConvNeXt family has four sizes: Tiny, Small, Base, and Large. `convnext_tiny` has around 28 million parameters and processes images through four stages with channel widths of 96, 192, 384, and 768. It is the lightest member of the family and the one we used. Pretrained ImageNet weights are available directly through `torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)`.

### Feature extraction

The model is not trained. The classification head is removed, and each image is passed through the backbone.

Processing steps:
1. Convolutional stages produce a spatial feature map
2. Global average pooling collapses spatial dimensions
3. LayerNorm + flatten produces a 768-dimensional vector

These vectors become inputs to downstream regressors. No gradients are computed. The backbone remains fixed.

## Why it fits this task

The images in the CSIRO Image2Biomass dataset are photographs of pasture plots taken under controlled field conditions. The targets dry green grass, dry dead grass, and dry clover correspond to visual texture, colour, and density cues that are distributed across each image rather than concentrated in a single salient object. Convolutional architectures are well suited to this because they build up local texture features at early layers and combine them into global statistics at later layers, which maps naturally onto the "count the grass" structure of the problem.

Using a pretrained backbone rather than training from scratch is essentially forced by the dataset size. With only around 357 training images (after the wide pivot), training a 28-million-parameter model from scratch would overfit catastrophically. ImageNet pretraining gives the backbone filters that already respond to textures, colours, and edges across a wide variety of natural images, so the 768-dimensional features it produces for a new image are already semantically rich even without any fine-tuning on biomass data.

ConvNeXt Tiny in particular was a good fit because it is small enough to run feature extraction quickly on CPU without requiring a GPU, while still producing richer features than ResNet-18, which is the other CNN baseline used in this project (ResNet-18 outputs 512-dimensional features via `model.fc = Identity()`, compared to ConvNeXt Tiny's 768-dimensional vectors).

## Implementation

Getting feature extraction running was straightforward compared to some of the other components in the pipeline. The main things worth noting:

**Loading the model.** `torchvision` ships ConvNeXt Tiny weights under `ConvNeXt_Tiny_Weights.DEFAULT`, which is the best available set of pretrained weights (ImageNet-1k). Loading is one line:

```python
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
```

**Stripping the head** The torchvision ConvNeXt Tiny `classifier` attribute is a `Sequential` of three layers: `LayerNorm`, `Flatten`, and then the final `Linear` projection to 1000 ImageNet classes. We want the 768-dimensional activations that come out of `LayerNorm` + `Flatten`, not the raw spatial map and not the class logits. The `ConvNeXtTinyBackbone` wrapper keeps `model.features` (the four convolutional stages), `model.avgpool` (global average pooling), and `model.pre_logits` (everything in the classifier except the last `Linear`) and discards only that final projection:

```python
class ConvNeXtTinyBackbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.pre_logits = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.pre_logits(x)
```

This is subtly different from just doing `backbone.classifier = nn.Identity()`, which would return the raw spatial tensor before pooling. By keeping `avgpool` and the `LayerNorm + Flatten` prefix of the classifier, we get properly normalised, flattened 768-d vectors ready to drop into a sklearn regressor.

**Freezing weights.** The model is put into eval mode with `.eval()` and all inference runs under `@torch.inference_mode()`, which disables gradient tracking entirely. There is no explicit `requires_grad_(False)` call because `inference_mode` makes it unnecessary.

**Batched inference.** Images are passed through the backbone in batches to keep memory usage reasonable. The output tensors are moved to CPU and collected into a single numpy array which is then used as the feature matrix `X` for the regressors.

**Preprocessing.** Images are resized to 350×700 (height × width) to preserve the 1:2 aspect ratio of the original field photographs. The normalisation uses the standard ImageNet mean and standard deviation (`[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`), consistent with what the pretrained weights expect. The `ImagePathDataset` class supports a `TRAIN` mode with random horizontal and vertical flips, but `img_data.py` always passes `mode=DataType.VAL` when calling it for feature extraction, so the deterministic transform is used in practice for both train and test images. The augmentation branch exists but is not exercised in the current pipeline.

**Feature caching.** Extracted features are saved to disk as `.npy` files (e.g. `feature_train_convnext.npy`) alongside a `.paths.txt` sidecar that records which image path corresponds to each row. On subsequent runs the pipeline checks for these files and skips extraction entirely if they already exist, which makes iterating on the regressor much faster.

**Sorted output.** After all batches are collected, the feature array is re-sorted by image path using `np.argsort` so the row order is deterministic regardless of DataLoader worker interleaving.

**PCA compression and the preprocessing pipeline.** The raw features are not passed directly to the regressor. Inside `baseline_training.py`, a `ColumnTransformer` wraps a `Pipeline([StandardScaler, PCA(n_components=128)])` over the numeric columns. The scaler centres and unit-normalises each feature dimension, then PCA reduces to 128 components. Crucially, this entire preprocessor is fit only on the training fold inside each CV split. This is the correct way to avoid leakage and is enforced by wrapping everything in a single sklearn `Pipeline` that is constructed fresh per fold by `model_wrapper_creator`.

The backbone wrapper lives in `src/main/vision/ConvNeXt/convnext.py`. The shared `ImagePathDataset` and `extract_features` loop live in `src/main/vision/dino.py`, which acts as the unified dispatcher: when `backbone="convnext"` it imports and calls `build_feature_extractor` from `convnext.py`, when `backbone="resnet"` it loads ResNet-18 with `model.fc = Identity()`, and when `backbone="dino"` it pulls DINOv2 from `torch.hub`. Switching backbones is entirely handled by the `--vision-backbone` CLI flag on both `run.py` and `run_only_train.py`.

## Results

### Cross-validation

The table below shows per-target R² under 5-fold GroupKFold cross-validation for both regressors with ConvNeXt Tiny features. Groups are plot IDs, so images from the same plot never appear in both the training fold and the validation fold. The model directly predicts three targets (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`); two further targets (`GDM_g = Dry_Green_g + Dry_Clover_g` and `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g`) are derived from those predictions inside the CV loop and included in the weighted R² metric. The Kaggle weights are 0.5 for `Dry_Total_g`, 0.2 for `GDM_g`, and 0.1 each for the three directly predicted targets.

| Regressor       | Dry\_Green\_g | Dry\_Dead\_g | Dry\_Clover\_g | Weighted avg |
| :-------------- | :-----------: | :----------: | :------------: | :----------: |
| ExtraTrees      |     0.754     |    0.186     |     0.342      |    0.620     |
| TabPFN          |     0.793     |    0.393     |     0.598      |    0.662     |

ConvNeXt is the strongest backbone on CV for both regressors, and the TabPFN + ConvNeXt combination is the best overall configuration we evaluated. The biggest per-target improvement over ResNet features is on `Dry_Dead_g` and `Dry_Clover_g`, which are the two harder targets. `Dry_Green_g` is already well predicted by all backbones, so the gains there are smaller.

Notably, ExtraTrees with ConvNeXt features achieves the same weighted R² as TabPFN with ResNet features (both 0.620), which suggests that upgrading the backbone and upgrading the regressor are roughly substitutable improvements, at least on CV.

### Kaggle leaderboard

| Regressor             | Private R² | Public R² |
| :-------------------- | :--------: | :-------: |
| ExtraTrees + ConvNeXt |   0.255    |   0.242   |
| TabPFN + ConvNeXt     |   0.232    |   0.193   |

On the private leaderboard ConvNeXt is the second-best backbone overall (ExtraTrees + DINO at 0.260 was the top entry), and ConvNeXt is the best backbone for the public leaderboard across both regressors. The reversal between TabPFN and ExtraTrees on Kaggle (TabPFN won on CV, ExtraTrees won on Kaggle) is discussed in the TabPFN writeup. It is likely a combination of the subsampled CV being optimistic and ExtraTrees generalising more conservatively on small noisy test sets.

The backbone ranking is more stable than the regressor ranking across CV and Kaggle: ConvNeXt is either first or second in every comparison, which gives some confidence that the ImageNet-pretrained features are genuinely more informative for this task than ResNet features.

## Reproducing

```bash
cd src

# CV evaluation with ConvNeXt features
uv run python -m main.run_only_train --model extra_trees --vision-backbone convnext
uv run python -m main.run_only_train --model tabpfn      --vision-backbone convnext

# Submission runs
uv run python -m main.run --model extra_trees --vision-backbone convnext
uv run python -m main.run --model tabpfn      --vision-backbone convnext
```

Feature extraction runs once and is cached.

I used the torchvision documentation and the original ConvNeXt paper to understand the architecture. AI was used to help proofread and structure this writeup.

## References

- Zhuang Liu et al. (2022). A ConvNet for the 2020s https://arxiv.org/abs/2201.03545
- torchvision ConvNeXt documentation https://pytorch.org/vision/stable/models/convnext.html
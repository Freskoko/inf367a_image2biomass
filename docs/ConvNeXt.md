# Novel Method: ConvNeXt (Vision Backbone)

## What it is

ConvNeXt is a pure convolutional neural network architecture designed to close the performance gap between CNNs and Vision Transformers (ViTs). It was introduced by Liu et al. (2022) in "A ConvNet for the 2020s". The key idea is that many of the design choices that made ViTs so successful — things like larger kernels, depthwise convolutions, fewer activation functions, and normalisation layers — can be transplanted back into a standard ResNet-style CNN. The result is a family of models that match or beat Swin Transformers on standard vision benchmarks while keeping the simplicity and efficiency of a convolutional architecture. For this project I used `convnext_tiny`, the smallest model in the family, loaded from `torchvision.models` with pretrained ImageNet weights, and used purely as a feature extractor.

## How it works

### Starting point: a modernised ResNet

The authors did not design ConvNeXt from scratch. Instead, they started from a standard ResNet-50 and applied a sequence of incremental changes, each one borrowed from a design decision that had been shown to help in the Transformer world. The result is a kind of roadmap from "classic CNN" to "what a CNN looks like if you take ViT ideas seriously".

The main changes in that roadmap are:

**Patchify stem.** The first layer is replaced with a 4×4 non-overlapping convolution with stride 4 — essentially the same as the patch embedding a ViT uses. This reduces the spatial resolution aggressively at the start instead of gradually.

**Depthwise convolutions with larger kernels.** Each residual block is restructured so that the expensive spatial mixing step uses a depthwise convolution (one filter per channel, no cross-channel mixing) with a 7×7 kernel. The cross-channel mixing is then done separately by pointwise (1×1) convolutions. The larger kernel means each neuron can see a bigger neighbourhood without a parameter explosion, since the depthwise structure keeps the filter count one-to-one with channels.

**Inverted bottleneck.** The hidden dimension inside each block is expanded, not contracted. In a standard ResNet bottleneck the 3×3 conv operates in a narrow bottleneck; in ConvNeXt the depthwise conv operates in the wide channel space and the pointwise projections handle the compression. This is the same structure used in MobileNets and also mirrors the MLP expansion inside a Transformer block.

**Fewer activation functions and normalisation layers.** A ResNet has a ReLU after every conv; ConvNeXt removes all but one activation per block and uses LayerNorm instead of BatchNorm. The LayerNorm placement and count more closely matches a Transformer, where normalisation happens once per block.

**Separate downsampling layers.** Between stages, ConvNeXt inserts explicit 2×2 stride-2 LayerNorm + conv layers for downsampling, rather than doing it inside the residual block. This matches the patch merging operation in Swin Transformer.

None of these changes is original in isolation, but putting them all together in a CNN produces a model that competes with Swin Transformer on ImageNet despite having no attention mechanism.

### The Tiny variant

The ConvNeXt family has four sizes: Tiny, Small, Base, and Large. `convnext_tiny` has around 28 million parameters and processes images through four stages with channel widths of 96, 192, 384, and 768. It is the lightest member of the family and the one we used. Pretrained ImageNet weights are available directly through `torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)`.

### Feature extraction

Because we are using ConvNeXt only as a feature extractor and not training it end-to-end, we strip off the classification head and forward each image through the convolutional backbone. After the four stages, the spatial feature map is averaged over its spatial dimensions with global average pooling, giving a single 768-dimensional vector per image. These vectors are what we feed into the downstream regressors (ExtraTrees and TabPFN). No gradients flow back into the convolutional weights at any point; the backbone is fully frozen.

## Why it fits this task

The images in the CSIRO Image2Biomass dataset are photographs of pasture plots taken under controlled field conditions. The targets — dry green grass, dry dead grass, and dry clover — correspond to visual texture, colour, and density cues that are distributed across each image rather than concentrated in a single salient object. Convolutional architectures are well suited to this because they build up local texture features at early layers and combine them into global statistics at later layers, which maps naturally onto the "count the grass" structure of the problem.

Using a pretrained backbone rather than training from scratch is essentially forced by the dataset size. With only around 357 training images (after the wide pivot), training a 28-million-parameter model from scratch would overfit catastrophically. ImageNet pretraining gives the backbone filters that already respond to textures, colours, and edges across a wide variety of natural images, so the 768-dimensional features it produces for a new image are already semantically rich even without any fine-tuning on biomass data.

ConvNeXt Tiny in particular was a good fit because it is small enough to run feature extraction quickly on CPU without requiring a GPU, while still producing richer features than ResNet-18, which is the other CNN baseline used in this project (ResNet-18 outputs 512-dimensional features via `model.fc = Identity()`, compared to ConvNeXt Tiny's 768-dimensional vectors).

## Implementation

Getting feature extraction running was straightforward compared to some of the other components in the pipeline. The main things worth noting:

**Loading the model.** `torchvision` ships ConvNeXt Tiny weights under `ConvNeXt_Tiny_Weights.DEFAULT`, which is the best available set of pretrained weights (ImageNet-1k). Loading is one line:

```python
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
```

**Stripping the head — carefully.** The torchvision ConvNeXt Tiny `classifier` attribute is a `Sequential` of three layers: `LayerNorm`, `Flatten`, and then the final `Linear` projection to 1000 ImageNet classes. We want the 768-dimensional activations that come out of `LayerNorm` + `Flatten`, not the raw spatial map and not the class logits. The `ConvNeXtTinyBackbone` wrapper keeps `model.features` (the four convolutional stages), `model.avgpool` (global average pooling), and `model.pre_logits` (everything in the classifier except the last `Linear`) and discards only that final projection:

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

**Freezing weights.** The model is put into eval mode with `.eval()` and all inference runs under `@torch.inference_mode()`, which disables gradient tracking entirely. There is no explicit `requires_grad_(False)` call because `inference_mode` makes it unnecessary — no computation graph is ever built.

**Batched inference.** Images are passed through the backbone in batches to keep memory usage reasonable. The output tensors are moved to CPU and collected into a single numpy array which is then used as the feature matrix `X` for the regressors.

**Preprocessing.** Images are resized to 350×700 (height × width) — twice as wide as tall — to preserve the 1:2 aspect ratio of the original field photographs. The normalisation uses the standard ImageNet mean and standard deviation (`[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`), consistent with what the pretrained weights expect. The `ImagePathDataset` class supports a `TRAIN` mode with random horizontal and vertical flips, but `img_data.py` always passes `mode=DataType.VAL` when calling it for feature extraction, so the deterministic transform is used in practice for both train and test images. The augmentation branch exists but is not exercised in the current pipeline.

**Feature caching.** Extracted features are saved to disk as `.npy` files (e.g. `feature_train_convnext.npy`) alongside a `.paths.txt` sidecar that records which image path corresponds to each row. On subsequent runs the pipeline checks for these files and skips extraction entirely if they already exist, which makes iterating on the regressor much faster.

**Sorted output.** After all batches are collected, the feature array is re-sorted by image path using `np.argsort` so the row order is deterministic regardless of DataLoader worker interleaving.

**PCA compression and the preprocessing pipeline.** The raw features are not passed directly to the regressor. Inside `baseline_training.py`, a `ColumnTransformer` wraps a `Pipeline([StandardScaler, PCA(n_components=128)])` over the numeric columns. The scaler centres and unit-normalises each feature dimension, then PCA reduces to 128 components. Crucially, this entire preprocessor is fit only on the training fold inside each CV split — the validation fold's statistics never touch the scaler or PCA basis. This is the correct way to avoid leakage and is enforced by wrapping everything in a single sklearn `Pipeline` that is constructed fresh per fold by `model_wrapper_creator`.

The backbone wrapper lives in `src/main/vision/ConvNeXt/convnext.py`. The shared `ImagePathDataset` and `extract_features` loop live in `src/main/vision/dino.py`, which acts as the unified dispatcher: when `backbone="convnext"` it imports and calls `build_feature_extractor` from `convnext.py`, when `backbone="resnet"` it loads ResNet-18 with `model.fc = Identity()`, and when `backbone="dino"` it pulls DINOv2 from `torch.hub`. Switching backbones is entirely handled by the `--vision-backbone` CLI flag on both `run.py` and `run_only_train.py`.

## Results

### Cross-validation

The table below shows per-target R² under 5-fold GroupKFold cross-validation for both regressors with ConvNeXt Tiny features. Groups are plot IDs, so images from the same plot never appear in both the training fold and the validation fold. The model directly predicts three targets (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`); two further targets (`GDM_g = Dry_Green_g + Dry_Clover_g` and `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g`) are derived from those predictions inside the CV loop and included in the weighted R² metric. The Kaggle weights are 0.5 for `Dry_Total_g`, 0.2 for `GDM_g`, and 0.1 each for the three directly predicted targets.

| Regressor       | Dry\_Green\_g | Dry\_Dead\_g | Dry\_Clover\_g | Weighted avg |
| :-------------- | :-----------: | :----------: | :------------: | :----------: |
| ExtraTrees      |     0.754     |    0.186     |     0.342      |    0.620     |
| TabPFN          |     0.793     |    0.393     |     0.598      |    0.662     |

ConvNeXt is the strongest backbone on CV for both regressors, and the TabPFN + ConvNeXt combination is the best overall configuration we evaluated. The biggest per-target improvement over ResNet features is on `Dry_Dead_g` and `Dry_Clover_g`, which are the two harder targets. `Dry_Green_g` is already well predicted by all backbones, so the gains there are smaller.

Notably, ExtraTrees with ConvNeXt features achieves the same weighted R² as TabPFN with ResNet features (both 0.620), which suggests that upgrading the backbone and upgrading the regressor are roughly substitutable improvements — at least on CV.

### Kaggle leaderboard

| Regressor             | Private R² | Public R² |
| :-------------------- | :--------: | :-------: |
| ExtraTrees + ConvNeXt |   0.255    |   0.242   |
| TabPFN + ConvNeXt     |   0.232    |   0.193   |

On the private leaderboard ConvNeXt is the second-best backbone overall (ExtraTrees + DINO at 0.260 was the top entry), and ConvNeXt is the best backbone for the public leaderboard across both regressors. The reversal between TabPFN and ExtraTrees on Kaggle (TabPFN won on CV, ExtraTrees won on Kaggle) is discussed in the TabPFN writeup — it is likely a combination of the subsampled CV being optimistic and ExtraTrees generalising more conservatively on small noisy test sets.

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

Feature extraction runs automatically on the first call and the `.npy` cache is reused on all subsequent runs.

I used the torchvision documentation and the original ConvNeXt paper to understand the architecture. AI was used to help proofread and structure this writeup.

## References

- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). A ConvNet for the 2020s. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://arxiv.org/abs/2201.03545
- PyTorch contributors. torchvision.models.convnext_tiny. https://pytorch.org/vision/stable/models/convnext.html
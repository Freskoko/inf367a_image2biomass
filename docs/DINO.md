# Novel Method: DINOv2 (Vision Backbone)

## What it is

DINOv2 is a Vision Transformer model developed by Meta AI. It builds on the original DINO method (“Distillation with No Labels”), where the model learns useful visual representations without using labelled data.

Instead of training on ImageNet classes, DINOv2 is trained in a self-supervised way. It learns by comparing different augmented versions of the same image and trying to produce consistent outputs. This forces the model to capture structure and patterns in the image rather than memorising labels.

In this project, we use `dinov2_vits14` from torch.hub as a frozen feature extractor.


## How it works

### Self-supervised training

The training process is based on a student–teacher setup:

- An image is transformed into multiple augmented views  
- A student network processes these views  
- A teacher network (a moving average of the student) produces target outputs  
- The student is trained to match the teacher  

This setup prevents collapse (same output for all inputs) and encourages the model to learn meaningful representations such as textures, shapes, and structure.


### Transformer backbone

DINOv2 uses a Vision Transformer (ViT):

- The image is split into patches (e.g. 14×14 pixels)  
- Each patch is treated as a token  
- Tokens are processed with self-attention  
- A CLS token aggregates global information  

Unlike CNNs, which build features locally, transformers model relationships across the entire image.

### The model used

We use:

- `dinov2_vits14`  
- ~21 million parameters  
- 384-dimensional output features  

Loaded with:

```python
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
```


### Feature extraction

The model is used in inference mode only.

Steps:
1. Image → patch embedding  
2. Transformer encoder  
3. CLS token extraction  
4. Output vector (384-dim)  

In some cases the model returns a dictionary:

```python
if isinstance(fb, dict):
    fb = fb["x_norm_clstoken"]
```

This is the feature used for downstream models.


## Why it fits this task

The CSIRO Image2Biomass dataset is not a typical object detection task. The targets depend on:

- texture  
- colour distribution  
- density  
- global structure  

These features are spread across the entire image rather than localized.

Transformers are well suited because they:
- capture global relationships  
- are not limited to local receptive fields  
- can model interactions between distant regions  


### Strengths

- strong pretrained representations  
- global context awareness  
- works well with small datasets  
- no need for fine-tuning  


### Weaknesses

- may lose fine local detail (CLS token limitation)  
- often benefits from fine-tuning (not done here)  
- less stable performance compared to CNNs in this setup  


## Implementation

### Model loading

```python
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval().to(device)
```


### Output handling

```python
if isinstance(fb, dict):
    fb = fb["x_norm_clstoken"]
```

### Shape validation

```python
if fb.ndim != 2:
    raise ValueError("Expected 2D feature tensor")
```



### Preprocessing

- Resize to `(image_size, image_size * 2)`  
- Normalize with ImageNet mean and std  
- No augmentation used during feature extraction  

### Batched inference

- DataLoader used for batching  
- Forward pass with `@torch.inference_mode()`  
- Features moved to CPU  
- Stored as numpy arrays  


### Sorting

```python
order = np.argsort(np.array(keys))
```

Ensures deterministic ordering of features.

### Feature caching

Saved as:

feature_train_dino.npy  
feature_test_dino.npy  
.paths.txt  

Prevents recomputation on repeated runs.


### Integration

DINO is integrated into the shared pipeline:

```python
build_feature_extractor(backbone="dino")
```

Selected via CLI:

```bash
--vision-backbone dino
```


## Results

### Cross-validation

DINO performed well but not best overall:

- slightly below ConvNeXt  
- similar on easier targets  
- weaker on more difficult targets  


| Regressor       | Dry_Green_g | Dry_Dead_g | Dry_Clover_g | Weighted avg |
| :-------------- | :---------: | :--------: | :----------: | :----------: |
| ExtraTrees      |   0.522     |   0.245    |    0.422     |    0.548     |
| TabPFN          |   0.668     |   0.459    |    0.555     |    0.677     |



### Kaggle leaderboard

| Model              | Private R² | Public R² |
|-------------------|------------|----------|
| ExtraTrees + DINO | 0.260      | 0.173    |
| TabPFN + DINO     | 0.118      | -0.174   |


### Interpretation

- Best private score overall (0.260)  
- Public score more unstable  
- Performance depends strongly on regressor  

Possible reasons:
- sensitivity to distribution shift  
- interaction between features and model  
- CV not fully representative of test data  


## Reproducing

```bash
cd src

# Cross-validation
uv run python -m main.run_only_train --model extra_trees --vision-backbone dino
uv run python -m main.run_only_train --model tabpfn      --vision-backbone dino

# Submission
uv run python -m main.run --model extra_trees --vision-backbone dino
uv run python -m main.run --model tabpfn      --vision-backbone dino
```

Feature extraction is cached and only runs once.


## References

- Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... & Bojanowski, P. (2023). Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
- PyTorch torch.hub DINOv2 implementation, https://github.com/facebookresearch/dinov2

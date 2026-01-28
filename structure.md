# Code setup

## Preproccessing

Scaling/PCA.. etc can be found [here](src/main/preprocessing)

## Vision

Vison models can be found [here](src/main/vision)

## Regression

Regression models can be found [here](src/main/regression)

# Pipeline:
    
1. Load data
- 1. IMAGES: augmentation, randomly flip training data
- 2. Tabular: No augmentation, flip to one row, only keep (clover, green and dead)

```text
        Total (Root)
       /            \
    Dead            GDM
                  /    \
              Green    Clover
```
<!-- source :https://www.kaggle.com/competitions/csiro-biomass/discussion/669012 -->

<!-- try dyno or some other image model -->
<!-- img to vector -->
2. Run vision on IMAGES (resnet)
- 1. Data saved in `src/main/model_data`

3. Preprocess
- 1. IMAGES: Run PCA on Resnet output data
- 2. TABULAR: No preprocessing

4. Combine data
- 1. Scale data (StandardScaler)

<!-- boosted models are best (randomforest) -->
<!-- confidence interval -->
5. Run multivariate regression model with 3 outputs
- 1. Test some baselines (KNN, SVR, ... etc)
- 2. Try more advanced model (Neural network)

6. Confidence of predictions



import numpy as np
import pandas as pd

# Load features
feats = np.load("features_train.npy")

# Load paths
with open("features_train.paths.txt", "r") as f:
    paths = f.read().splitlines()

# Create feature column names
feat_cols = [f"feature_{i}" for i in range(feats.shape[1])]

# Create dataframe
df = pd.DataFrame(feats, columns=feat_cols)
df.insert(0, "image_path", paths)

# Save to CSV
df.to_csv("features_train.csv", index=False)

print("Saved features_train.csv")
print(df.head())

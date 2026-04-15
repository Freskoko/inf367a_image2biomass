from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

OUTPUT_NAME = "BIOMASS_64x64" # biomass
DATA_DIR = "src/data"

# --- SETTINGS ---
CSV_PATH = Path(f"{DATA_DIR}/train.csv")
IMG_DIR = Path(F"{DATA_DIR}")
OUTPUT_H5 = Path(f"{DATA_DIR}/{OUTPUT_NAME}.h5")
IMG_SIZE = 64
PRIMARY_TARGET = "Dry_Total_g"

# 1. pivot and floor
print("Pivoting and flooring dataset...")
df_long = pd.read_csv(CSV_PATH)
df_wide = df_long.pivot_table(
    index=["image_path"], columns="target_name", values="target"
).reset_index()

biomass_cols = df_wide.select_dtypes(include=[np.number]).columns
df_wide[biomass_cols] = np.floor(df_wide[biomass_cols])

# 2. process
imgs_list = []
labels_list = []

print(f"Packing images...")
for _, row in tqdm(df_wide.iterrows(), total=len(df_wide)):
    img_full_path = IMG_DIR / row["image_path"]

    if not img_full_path.exists():
        print(f"warning cannot find image {img_full_path}")
        continue

    img = Image.open(img_full_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    img_array = np.array(img).transpose(2, 0, 1)
    imgs_list.append(img_array)
    labels_list.append(row[PRIMARY_TARGET])

# Convert to numpy arrays
imgs_all = np.array(imgs_list).astype(np.uint8)
labels_all = np.array(labels_list).astype(np.float32)

# --- NEW: STEP 3. Split into Train and Test (Validation) ---
# We'll use 80% for training and 20% for the evaluation CNN to check its homework.
print("Splitting data into train/test sets...")
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    imgs_all, labels_all, test_size=0.2, random_state=2020
)

# 4. Save to H5 with the names the CcGAN repo expects
with h5py.File(OUTPUT_H5, "w") as f:
    # Training set
    # f.create_dataset('images', data=train_imgs)
    # f.create_dataset('labels', data=train_labels)

    f.create_dataset("imgs_color", data=train_imgs)
    f.create_dataset("cell_counts", data=train_labels)
    # Validation/Test set
    # f.create_dataset('labels', data=test_labels)
    f.create_dataset("imgs_color_test", data=train_imgs)
    f.create_dataset("cell_counts_test", data=train_labels)

print(f"\nSuccess! Created {OUTPUT_H5}")
print(f"Train samples: {len(train_labels)}")
print(f"Test samples: {len(test_labels)}")
print(f"Max label: {np.max(labels_all)} | Min label: {np.min(labels_all)}")

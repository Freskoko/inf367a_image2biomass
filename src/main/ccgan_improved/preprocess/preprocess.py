from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

"""File to preprocess image2biomass data to appropriate form for the CcGAN method"""


OUTPUT_NAME = "BIOMASS_64x64"
DATA_DIR = "src/data"

CSV_PATH = Path(f"{DATA_DIR}/train.csv")
IMG_DIR = Path(f"{DATA_DIR}")
OUTPUT_H5 = Path(f"{DATA_DIR}/{OUTPUT_NAME}.h5")
IMG_SIZE = 64
PRIMARY_TARGET = "Dry_Total_g"  # chosen variable

print("pivoting and flooring dataset...")
df_long = pd.read_csv(CSV_PATH)
df_wide = df_long.pivot_table(
    index=["image_path"], columns="target_name", values="target"
).reset_index()

biomass_cols = df_wide.select_dtypes(include=[np.number]).columns
df_wide[biomass_cols] = np.floor(df_wide[biomass_cols])

imgs_list = []
labels_list = []

print(f"packing images...")
for _, row in tqdm(df_wide.iterrows(), total=len(df_wide)):
    img_full_path = IMG_DIR / row["image_path"]

    if not img_full_path.exists():
        print(f"warning! cannot find image {img_full_path}")
        continue

    img = Image.open(img_full_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    img_array = np.array(img).transpose(2, 0, 1)
    imgs_list.append(img_array)
    labels_list.append(row[PRIMARY_TARGET])

imgs_all = np.array(imgs_list).astype(np.uint8)
labels_all = np.array(labels_list).astype(np.float32)

# 80% train, 20% test
print("splitting data into train/test sets...")
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    imgs_all, labels_all, test_size=0.2, random_state=2026
)

with h5py.File(OUTPUT_H5, "w") as f:
    # train
    f.create_dataset("imgs_color", data=train_imgs)
    f.create_dataset("cell_counts", data=train_labels)
    # test
    f.create_dataset("imgs_color_test", data=test_imgs)
    f.create_dataset("cell_counts_test", data=test_labels)

print(f"Created {OUTPUT_H5}")
print(f"train samples len: {len(train_labels)}")
print(f"test samples len: {len(test_labels)}")
print(f"max label: {np.max(labels_all)} min label: {np.min(labels_all)}")
# max and min are settings which must be set in run_train.sh

# ](../src/data/train/ID1103883611.jpg
from pathlib import Path
from PIL import Image

DATA_DIR = "src/data"
IMG_FILE = Path(F"{DATA_DIR}/train/ID1103883611.jpg")
IMG_SIZE = 64

img = Image.open(IMG_FILE).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
img.save("docs/images/compressed_95.jpg")

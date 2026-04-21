from pathlib import Path
from PIL import Image

"""Quick helper to compress one image to 64x64"""

DATA_DIR = "src/data"
IMG_FILE = Path(f"{DATA_DIR}/train/ID1103883611.jpg")
IMG_SIZE = 64

img = Image.open(IMG_FILE).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
img.save("docs/images/compressed_95.jpg")

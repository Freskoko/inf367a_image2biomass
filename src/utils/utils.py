from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def load_one_image(img_id):
    """
    Loads an image from training data
    """
    path = Path(f"data/train/{img_id}.jpg")
    return Image.open(path).convert("RGB")

def plot_one_image(img_id):
    img = load_one_image(img_id)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Image {img_id}")
    plt.show()

if __name__ == "__main__":
    plot_one_image("ID4464212")
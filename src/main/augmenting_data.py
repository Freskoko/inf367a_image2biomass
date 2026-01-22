from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2

# ideas from here:
# https://link.springer.com/article/10.1186/s40537-019-0197-0?

@dataclass(frozen=True)
class AugmentingConfig:
    image_standard = (2000,1000) # width,height
    gauss_std = 1

# def create_random_subset():

# def create_new_data():
#     return ...

# https://www.askpython.com/python/examples/adding-noise-images-opencv
def add_gaussian_noise(image, cfg):
    img = cv2.imread(image)
    noise = np.random.normal(0, cfg.gauss_std, img.shape).astype(np.uint8)
    noisy_image = cv2.add(img, noise)
    # cv2.imwrite("gaussian.png",noisy_image)
    return noisy_image


def flip_image(image_path):
    """
    Horizontally flips image
    """
    img = Image.open(image_path)
    vertical = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    # vertical.save("vertical.png")
    return vertical

def kernel_filters():
    ...


def colour_casting():
    ...
    # colour aguments

def edge_enhancment():
    ...


if __name__ == "__main__":
    cfg = AugmentingConfig()
    img = "src/data/train/ID4464212.jpg"
    # flip_image(img)
    add_gaussian_noise(img, cfg)




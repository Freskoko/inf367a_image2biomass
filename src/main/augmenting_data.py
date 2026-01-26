from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from torchvision import transforms

random.seed(367)

IMG_HEIGHT = 385
IMG_WIDTH = IMG_HEIGHT * 2

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
def add_gaussian_noise(img_id, cfg):
    """
    Adds gaussian noise, strength determined by config
    """
    img = cv2.imread(f"data/train/{img_id}")
    noise = np.random.normal(0, cfg.gauss_std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def flip_image(img_id):
    """
    Horizontally flips image
    """
    img = cv2.imread(f"data/train/{img_id}")
    return cv2.flip(img, 1)
    

def kernel_filters():
    ...


def colour_casting():
    ...
    # colour aguments


def edge_enhancment():
    ...

def save_img(img, hash):
    cv2.imwrite(f"data/interim/augmented/{hash}.jpg", img)

def create_csv_copy(train_id, new_hash):
    df = pd.read_csv("data/train.csv")
    original = df[df['sample_id'].str.contains(train_id)]
    original["image_path"] = f"data/interim/augmented/{new_hash}.jpg"
    original["sample_id"] = original["sample_id"].str.replace(train_id,new_hash)
    return original
    
def gan():
    ...
    # maybe?


def get_train_transform():
    """
    When training across epochs,
    images are randomly flipped, either vertically, horizontally, or both
    Gaussian noise is also added
    """
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.v2.GaussianNoise(mean = 1, std = 0.1),
        transforms.ToTensor(),
        
        # normalize values are chosen based on backbone model
        # channel-wise mean and std of imagenet
        # make images look statistically like imagenet images
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """Validation transforms without augmentation"""
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    cfg = AugmentingConfig()
    img_id = "ID4464212"
    hash = f"ID{random.getrandbits(28)}"

    df = create_csv_copy(img_id, hash)
    print(df)

    # img = flip_image(img_id)
    # save_img(img)
    # add_gaussian_noise(img, cfg)


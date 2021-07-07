from torch.utils import data
from torchvision import datasets, transforms
import torch
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def pil_loader(path, img_size=224):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except FileNotFoundError as e:
        raise FileNotFoundError(e)

 
def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [64.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.Resize(256))
    if random_crop:
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomResizedCrop(224)
    else:
        transform.append(transforms.CenterCrop(224))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)




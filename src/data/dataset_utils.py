from PIL import Image

from hydra.utils import instantiate
from omegaconf import OmegaConf

import numpy as np


def read_txt_by_lines(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = list(map(lambda s: s[:-1], data))
    
    return data


def read_grayscale_uint8(path, normalize=True):
    image = np.array(Image.open(path))
    image = image / 255 if normalize else image
    image = image.astype(np.float32)

    return image


def exclude_random_transforms(transforms):
    """Return initial list with elements that are always applied.
    """
    new_transforms = []
    for i, transform in enumerate(transforms):
        if OmegaConf.is_dict(transform):
            transform = instantiate(transform, _convert_="partial")
        if transform.always_apply:
            new_transforms.append(transforms[i])
    
    return new_transforms

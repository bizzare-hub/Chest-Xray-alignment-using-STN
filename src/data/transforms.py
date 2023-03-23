import numpy as np

from albumentations import ImageOnlyTransform


class GrayToRGB(ImageOnlyTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.ndim not in [2, 3]:
            msg = f"Wrong number of dims: {img.ndim} "
            msg += "Only images of format (H, W) or (H, W, 1) are supported"
            raise ValueError(msg)
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        if img.shape[-1] == 4:  # alpha channel
            img = img[:, :, :-1] 
        if img.shape[-1] not in [1, 3]:
            msg = f"Wrong number of channels: {img.shape[-1]}"
            msg += "Only 1 or 3 channels are supported"
            raise ValueError(msg)
        
        img = img if img.shape[-1] == 3 \
            else np.tile(img, reps=[1, 1, 3])
        
        return img
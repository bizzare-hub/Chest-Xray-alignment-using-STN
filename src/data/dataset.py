import logging
from typing import List, Union, Callable
from pathlib import Path

import pandas as pd

import torch
import albumentations as A

from .dataset_utils import (
    read_txt_by_lines, read_grayscale_uint8
)


logger = logging.getLogger(__name__)


PathType = Union[str, Path]


class ChestXRayAlignmentDataset:
    def __init__(
        self,
        main_dir: PathType,
        transforms: List[Callable],
        anchor_transforms: List[Callable] = None,
        images: str = '',
        annot: str = '',
        anchor: str = None,
        split: str = None
    ) -> None:
        """Base dataset class for ChestXRay classification datasets.

        Args:
            main_dir (PathType): directory where all data is stored. That
                includes folders with images, annotations, etc.
            transforms (List[TransformsSeqType]): list of albumentations transforms 
                to apply on the image
            images (str): subdirectory of main_dir where images are located
            anchor (str): name of .png image - canonical chest
              (located in main_dir).
            annot (str): subdirectory of main_dir where (usually) pd.DataFrame
                with labels and relative image paths are located
            split (str): path to .txt that contains list of subject ids to use.
                If None (default), take all.
		"""
        self.main_dir = Path(main_dir)
        
        self.pipeline = A.Compose(transforms)
        self.anchor_pipeline = A.Compose(anchor_transforms)
        self.split = split

        self.load_annotations(self.main_dir / images, self.main_dir / annot)
        self._compute_anchor_chest(anchor)

    def load_annotations(self, images_dir, annot_dir) -> None:
        records = pd.read_csv(annot_dir / "annotations.csv")
        if self.split is not None:
            subjects = read_txt_by_lines(annot_dir / (self.split + ".txt"))
            records = records[records.subject_id.astype(str).isin(subjects)]

        self._paths = records.path.apply(lambda p: str(images_dir / p)).values

        self._n_samples = len(self._paths)

        logger.info(f"Dataset size: {self._n_samples}")
    
    def _compute_anchor_chest(self, name) -> None:
        if name is None:
            self.anchor = None
        else:
            path = self.main_dir / (name + ".png")

            anchor = read_grayscale_uint8(path, normalize=True)
            anchor = self.anchor_pipeline(image=anchor)["image"] 
            self.anchor = anchor.to(torch.float32)
            logger.info(f"Prepared canonical chest from : {path}")

    def prepare_data(self, idx):
        path = self._paths[idx]

        image = read_grayscale_uint8(path, normalize=True)
        image = self.pipeline(image=image)["image"]
        image = image.to(torch.float32)

        return image
    
    def __getitem__(self, idx):
        """
        Caution: don't use standard python containers (list, dict) here.
          Instead use numpy.
        Why? ---> https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        """
        return self.prepare_data(idx)

    def __len__(self):
        return self._n_samples

import logging
from typing import Union
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from .data.dataset import ChestXRayAlignmentDataset


logger = logging.getLogger(__name__)


ConfigType = Union[dict, DictConfig]


class PLAlignmentModel(LightningModule):

    _train_dataset: ChestXRayAlignmentDataset = None

    def __init__(
        self,
        model_config: ConfigType,
        loader_config: ConfigType = None,
        loss_config: ConfigType = None,
        optimizer_config: ConfigType = None,
        lr_scheduler_config: ConfigType = None,
        log_every_n_steps: int = None
    ) -> None:
        super().__init__()

        self._model: nn.Module = instantiate(model_config.module, _convert_="partial")

        self._loss = instantiate(loss_config, _convert_="partial")
    
        self._loader_config = loader_config
        self._optimizer_config = optimizer_config
        self._lr_scheduler_config = lr_scheduler_config
        self.log_every_n_steps = log_every_n_steps

    @property
    def train_dataset(self):
        return self._train_dataset
  
    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset
    
    def forward(self, x):
        return self._model(x)

    def on_train_start(self) -> None:
        logger.info("Calculating anchor features: Start")
        
        anchor = self.train_dataset.anchor[None]
        anchor = anchor.to(self.device)
        self._anchor_features = [anchor] + self._model.extract_features(anchor)

        # anchor_features = [0]
        # n_samples = 1000
        # for idx in range(n_samples):
        #     image = self.train_dataset[idx]
        #     image = F.interpolate(image[None], size=[320, 256])
        #     features = self._model.extract_features(image.to(self.device))
        #     anchor_features = [anchor_feat + (feat / n_samples)
        #                        for anchor_feat, feat in zip(anchor_features, features)]
        # self._anchor_features = [anchor] + anchor_features

        logger.info("Calculating anchor features: Done")
    
    def training_step(self, batch, batch_idx) -> None:
        images = batch
        features = self(images)
        
        loss = self._loss(features, self._anchor_features)

        if self.log_every_n_steps is not None and ((batch_idx + 1) % self.log_every_n_steps) == 0:
            self._log_random_pair(images, features[0])        

        return {"loss": loss}
    
    @rank_zero_only
    def _log_random_pair(self, batch, batch_transform) -> None:
        batch_size, _, h, w = batch_transform.size()
        batch = F.interpolate(batch, size=[h, w])

        idx = torch.randint(high=batch_size, size=(1, )).item()

        x, xt = batch[idx][0].unsqueeze(0), batch_transform[idx][0].unsqueeze(0)
        grid = make_grid(torch.stack([x, xt, self.train_dataset.anchor[0].unsqueeze(0).to(self.device)]), nrow=2)

        self.logger.experiment.add_image(f"{self.global_step}", grid)

    def configure_optimizers(self):
        optimizer = instantiate(self._optimizer_config, params=self._model.parameters(), _convert_="partial")
        if self._lr_scheduler_config is not None:
            scheduler = instantiate(self._lr_scheduler_config, optimizer=optimizer, _convert_="partial")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self._loader_config)

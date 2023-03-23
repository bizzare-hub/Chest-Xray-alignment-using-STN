import sys
import logging
import os.path as osp

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

import cv2

from pytorch_lightning.callbacks import (
    ModelCheckpoint, RichProgressBar, LearningRateMonitor)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from .models import PLAlignmentModel


stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers
)

logger = logging.getLogger(__name__)


def multiple_gpu_setup(cfg: DictConfig) -> None:
    """Setup multigpu pytorch w. opencv compatibility.

    For more info refer to: https://github.com/albumentations-team/albumentations,
      ``Comments`` section.
    """
    n_devices = cfg.devices

    if OmegaConf.is_list(n_devices):
        n_devices = len(n_devices)
    if n_devices > 1:
        msg = f"Number of devices = {n_devices} > 1. "
        msg += "Setting n_threads = 0 and disabling opencl "
        msg += "to prevent deadlocks."
        logger.info(msg)

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)


def train(cfg: DictConfig):
    multiple_gpu_setup(cfg.trainer)

    experiment_step = cfg.get("experiment_step", "")

    ckpt_dir = osp.join(cfg.experiment_path, "checkpoints", cfg.experiment_name, experiment_step)
    logs_dir = osp.join(cfg.experiment_path, "logs", cfg.experiment_name, experiment_step)
    
    train_dataset = instantiate(cfg.data.dataset, _convert_="partial")

    model = PLAlignmentModel(
        cfg.model,
        cfg.data.loader,
        cfg.loss,
        cfg.optimizer,
        cfg.lr_scheduler,
        log_every_n_steps=250
    )

    model.train_dataset = train_dataset

    callbacks = [
        ModelCheckpoint(
            ckpt_dir,
            filename="{epoch}",
            every_n_epochs=1),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar()
    ]

    pl_logger = [TensorBoardLogger(save_dir=logs_dir)]

    trainer = Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        logger=pl_logger,
        **cfg.trainer,
    )

    trainer.fit(model)

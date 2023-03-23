import logging
from typing import Optional, Dict, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


logger = logging.getLogger(__name__)


class TimmClassificationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = False,
        num_classes: int = 14,
        global_pool: str = 'avg',
        drop_rate: float = 0.,
        checkpoint_path: str = None,
    ):
        super().__init__()

        if backbone not in timm.list_models():
            raise ValueError(f"No {backbone} backbone in timm library.")
        
        self.model = timm.create_model(
            backbone, pretrained=pretrained, 
            num_classes=num_classes, global_pool=global_pool, drop_rate=drop_rate)

        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            self._load_from_checkpoint(checkpoint_path)
        
    def forward(self, x):
        return self.model(x)

    def _load_from_checkpoint(self, ckpt_path):
        def remove_prefix(s: str, prefix: str):
            if s.startswith(prefix):
                s = s[len(prefix):]
            
            return s
        
        checkpoint = torch.load(ckpt_path)
        state_dict = checkpoint["state_dict"]
        
        ex_name = next(iter(state_dict.keys()))
        rm_pattern = ex_name[:ex_name.find("features")]
        
        state_dict = OrderedDict({remove_prefix(key, f"{rm_pattern}"): val
                                  for key, val in state_dict.items()})
        
        self.model.load_state_dict(state_dict)


class SpatialTransformBlock(nn.Sequential):
    def __init__(self, in_channels: int, hid_channels: int):
        super().__init__()

        self.add_module("fc1", nn.Linear(in_channels, hid_channels))
        self.add_module("act1", nn.ReLU())
        self.add_module("fc2", nn.Linear(hid_channels, 5))
        # self.add_module("fc2", nn.Linear(hid_channels, 6))

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(
            torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32))
        # self.fc2.bias.data.copy_(
        #     torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
    
    def forward(self, x):
        x = super().forward(x)

        theta = torch.stack(
            [x[:, 2] * torch.cos(x[:, 4]), -x[:, 3] * torch.sin(x[:, 4]), x[:, 0],
             x[:, 2] * torch.sin(x[:, 4]), x[:, 3] * torch.cos(x[:, 4]), x[:, 1]],
            dim=1
        )

        return theta


class AlignmentModel(nn.Module):
    _model2features = {
        'vgg16': ('15'),
        'densenet121': ('transition2')
    }

    def __init__(
        self,
        backbone: str,
        loss_network: str,
        n_backbone_out_channels: int = 512,
        grid_shape: List[int] = [3, 320, 256],
        backbone_kwargs: Optional[Dict] = None,
        loss_network_kwargs: Optional[Dict] = None
    ):
        super().__init__()

        if loss_network not in self._model2features:
            raise ValueError(f"{loss_network} not supported as loss network.")

        self.loss_net_features \
            = self._model2features[loss_network]
        self.grid_shape = grid_shape

        self.backbone = TimmClassificationModel(
            backbone=backbone, **backbone_kwargs)
        self.loss_network = TimmClassificationModel(
            backbone=loss_network, **loss_network_kwargs)
        
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.stn = SpatialTransformBlock(
            n_backbone_out_channels, n_backbone_out_channels // 2)
        
    def forward(self, x):
        features = self.backbone(x)

        theta = self.stn(features)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, [theta.size(0)] + self.grid_shape)
        x = F.grid_sample(x, grid)

        perceptual_features = self.extract_features(x)

        if self.training:
            return [x] + perceptual_features
        else:
            return x

    def extract_features(self, x):
        self.loss_network.eval()

        features = []
        for name, layer in self.loss_network.model.features.named_children():
            x = layer(x)
            if name in self.loss_net_features:
                features.append(x)
            if len(features) >= len(self.loss_net_features):
                break
    
        return features

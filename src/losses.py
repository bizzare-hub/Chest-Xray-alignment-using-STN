import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MSELoss


def perceptual_loss(input, target):
    """Perceptual Loss as in
      https://arxiv.org/pdf/1603.08155.pdf
    
    Args:
      input (List[torch.Tensor]): List of features maps
        from different layers of loss_network
        (can also include input image x).
      target (List[torch.Tensor]): List of target-style
        features maps, same as for input.
    """
    batch_size = input[0].size(0)

    losses = []
    for idx, (inp, tgt) in enumerate(zip(input, target)):
        if tgt.size(0) == 1:
            tgt = tgt.repeat([batch_size, 1, 1, 1])

        loss = F.mse_loss(inp, tgt)
        if idx == 0:
            loss = 15.0 * loss
        losses.append(loss)

    losses = torch.stack(losses)

    return torch.mean(losses)


class PerceptualLoss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return perceptual_loss(input, target)

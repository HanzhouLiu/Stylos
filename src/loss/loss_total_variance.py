from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss, T_wrapper





@dataclass
class LossTotalVarianceCfg:
    weight: float
    conf: bool = False
    mask: bool = False
    alpha: bool = False

@dataclass
class LossTotalVarianceCfgWrapper:
    total_variance: LossTotalVarianceCfg


class TVLoss(nn.Module):
    """L2 total variation loss (nine point stencil)."""

    def forward(self, input):
        input = F.pad(input, (1, 1, 1, 1), 'replicate')
        s1, s2 = slice(1, -1), slice(2, None)
        s3, s4 = slice(None, -1), slice(1, None)
        d1 = (input[..., s1, s2] - input[..., s1, s1]).pow(2).mean() / 3
        d2 = (input[..., s2, s1] - input[..., s1, s1]).pow(2).mean() / 3
        d3 = (input[..., s4, s4] - input[..., s3, s3]).pow(2).mean() / 12
        d4 = (input[..., s4, s3] - input[..., s3, s4]).pow(2).mean() / 12
        return 2 * (d1 + d2 + d3 + d4)
    
class LossTotalVariance(Loss[LossTotalVarianceCfg, LossTotalVarianceCfgWrapper]):

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)
        self.tv_loss_fn = TVLoss()

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Get alpha and valid mask from inputs
        alpha = prediction.alpha
        # valid_mask = torch.ones_like(alpha, device=alpha.device).bool()
        valid_mask = batch['context']['valid_mask']

        # # only for objaverse
        # if batch['context']['valid_mask'].sum() > 0:
        #     valid_mask = batch['context']['valid_mask']

        # Determine which mask to use based on config
        if self.cfg.mask:
            mask = valid_mask
        elif self.cfg.alpha:
            mask = alpha  
        elif self.cfg.conf:
            mask = depth_dict['conf_valid_mask']
        else:
            mask = torch.ones_like(alpha, device=alpha.device).bool()

        # Rearrange and mask predicted and ground truth images
        if mask.shape != prediction.color.shape:
            mask = mask.unsqueeze(2).expand_as(prediction.color)
        rendered_img = prediction.color*mask
        B, S, C, H, W = rendered_img.shape

        rendered_img = rendered_img.view(B * S, C, H, W)

        tv_loss = self.tv_loss_fn(rendered_img)
        return tv_loss * self.cfg.weight
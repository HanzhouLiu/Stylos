from dataclasses import dataclass
from typing import List

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss, T_cfg, T_wrapper, ScaledMSELoss, VGG19Features

class ContentLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss = ScaledMSELoss(eps=eps)

    def forward(self, input, target):
        return self.loss(input, target)


class ContentLossMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target)

@dataclass
class LossContentCfg:
    weight: float
    conf: bool = False
    mask: bool = False
    alpha: bool = False
    use_checkpoint: bool = False

@dataclass
class LossContentCfgWrapper:
    content: LossContentCfg


class LossContent(Loss[LossContentCfg, LossContentCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)
        self.layers = [20, 29]
        self.vgg = VGG19Features(layers=self.layers)
        self.loss_fn = ContentLoss()


    def _forward(
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
        target_img = ((batch["context"]["image"][:, batch["using_index"]] + 1) / 2)*mask
        B, S, C, H, W = rendered_img.shape

        rendered_img = rendered_img.view(B * S, C, H, W)
        target_img = target_img.view(B * S, C, H, W)
        # Extract features
        rendered_feats = self.vgg(rendered_img)
        target_feats   = self.vgg(target_img)
        # Content Loss (feature differences)
        content_loss = 0.0
        for layer in self.layers:
            content_loss += self.loss_fn(rendered_feats[layer], target_feats[layer])
        del rendered_feats, target_feats
        return content_loss * self.cfg.weight

    def forward(self, prediction, batch, gaussians, depth_dict, global_step):
        if self.cfg.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, prediction, batch, gaussians, depth_dict, global_step
            )
        else:
            return self._forward(prediction, batch, gaussians, depth_dict, global_step)

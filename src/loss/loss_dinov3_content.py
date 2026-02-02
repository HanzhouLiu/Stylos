from dataclasses import dataclass
from typing import List

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss, T_cfg, T_wrapper, ScaledMSELoss, VGG19Features

class DINOV3ContentLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss = ScaledMSELoss(eps=eps)

    def forward(self, input, target):
        return self.loss(input, target)


class DINOV3ContentLossMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target)

@dataclass
class LossDINOV3ContentCfg:
    weight: float
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    conf: bool = False
    mask: bool = False
    alpha: bool = False

@dataclass
class LossDINOV3ContentCfgWrapper:
    dinov3content: LossDINOV3ContentCfg



PATCH_SIZE = 16
IMAGE_SIZE = 224
class LossDINOV3Content(Loss[LossDINOV3ContentCfg, LossDINOV3ContentCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.dinov3_model = AutoModel.from_pretrained(self.cfg.model_name)
        self.dinov3_model.eval()
        self.dinov3_model.requires_grad_(False)
        self.loss_fn = DINOV3ContentLoss()

    def resize_transform(
        self,
        images: Tensor,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
    ) -> torch.Tensor:
        B , C, H, W= images.shape
        h_patches = int(H / patch_size)
        w_patches = int((W * image_size) / (H * patch_size))
        return self.normalize(TF.resize(images, (h_patches * patch_size, w_patches * patch_size)))

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
        rendered_img = self.resize_transform(rendered_img)
        target_img = self.resize_transform(target_img)
        # Content Loss (feature differences)
        rendered_feats = self.dinov3_model(pixel_values=rendered_img).last_hidden_state
        rendered_feats = rendered_feats[:, 1 + self.dinov3_model.config.num_register_tokens:]
        target_feats = self.dinov3_model(pixel_values=target_img).last_hidden_state
        target_feats = target_feats[:, 1 + self.dinov3_model.config.num_register_tokens:].detach()
        content_loss = self.loss_fn(rendered_feats, target_feats)
        del rendered_feats, target_feats
        return content_loss * self.cfg.weight

    def forward(self, prediction, batch, gaussians, depth_dict, global_step):
        return self._forward(prediction, batch, gaussians, depth_dict, global_step)

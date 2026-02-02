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
from .loss import Loss, T_cfg, T_wrapper



class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features.eval()
        for param in vgg_pretrained.parameters():
            param.requires_grad = False

        self.layers = {
            'relu1_1': 1,
            'relu2_1': 6,
            'relu3_1': 11,
            'relu4_1': 20
        }

        self.vgg = vgg_pretrained

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            for key, idx in self.layers.items():
                if int(name) == idx:
                    features[key] = x
        return features


@dataclass
class LossStyleContentCfg:
    style_weight: float = 1.0
    content_weight: float = 1.0

@dataclass
class LossStyleContentCfgWrapper:
    style_content: LossStyleContentCfg


class LossStyleContent(Loss[LossStyleContentCfg, LossStyleContentCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)
        self.vgg = VGG19Features()
        
    def compute_mean_variance(self, features):
        mean = features.mean(dim=[2, 3], keepdim=True)
        var = features.var(dim=[2, 3], keepdim=True)
        return mean, var

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
        target_img = ((batch["context"]["image"][:, batch["using_index"]] + 1) / 2)*mask
        B, S, C, H, W = rendered_img.shape

        style_img = ((batch["context"]["style_image"] + 1) / 2)

        rendered_img = rendered_img.view(B * S, C, H, W)
        target_img = target_img.view(B * S, C, H, W)
        style_img = ((batch["context"]["style_image"] + 1) / 2).view(B * 1, C, H, W)
        # Extract features
        rendered_feats = self.vgg(rendered_img)
        style_feats    = self.vgg(style_img)
        target_feats   = self.vgg(target_img)

        # Style Loss (mean/variance differences)
        style_loss = 0.0
        for layer in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']:
            mean_r, var_r = self.compute_mean_variance(rendered_feats[layer])
            _,C_s, fh, fw = mean_r.shape
            mean_r = mean_r.view(B, S, C_s, fh, fw)
            var_r = var_r.view(B, S, C_s, fh, fw)
            mean_s, var_s = self.compute_mean_variance(style_feats[layer])
            mean_s = mean_s.view(B, 1, C_s, fh, fw)
            var_s = var_s.view(B, 1, C_s, fh, fw)
            mean_s = mean_s.expand_as(mean_r)
            var_s = var_s.expand_as(var_r)

            style_loss += F.mse_loss(mean_r, mean_s) + F.mse_loss(var_r, var_s)

        # Content Loss (feature differences)
        content_loss = 0.0
        for layer in ['relu3_1', 'relu4_1']:
            content_loss += F.mse_loss(rendered_feats[layer], target_feats[layer])

        # Weighted sum
        total_loss = self.style_weight * style_loss + self.content_weight * content_loss

        return total_loss

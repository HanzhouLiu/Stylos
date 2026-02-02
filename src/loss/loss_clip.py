from dataclasses import dataclass
from functools import partial
from typing import List

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import open_clip

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss, T_cfg, T_wrapper, ScaledMSELoss, VGG19Features
from .sqrtm import sqrtm_ns_lyap



@dataclass
class LossCLIPCfg:
    weight: float = 1.0
    n_cuts: int = 4
    cut_target: bool = True  # If True, cut the target image into patches


@dataclass
class LossCLIPCfgWrapper:
    clip: LossCLIPCfg


class CLIPLoss(torch.nn.Module):
  def __init__(self, text_prompts=[], image_prompts=[], n_cuts=16):
    super(CLIPLoss, self).__init__()
    
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    self.clip_model = clip_model
    self.clip_model_input_size = 224
    self.preprocess = transforms.Compose([
        transforms.Resize(size=self.clip_model_input_size, max_size=None, antialias=None),
        transforms.CenterCrop(size=(self.clip_model_input_size, self.clip_model_input_size)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    self.clip_model.to('cuda')
    self.clip_model.eval()
    
    self.target_embeds = []
    with torch.no_grad():
      for text_prompt in text_prompts:
        tokenized_text = open_clip.tokenize([text_prompt]).to('cuda')
        self.target_embeds.append(clip_model.encode_text(tokenized_text))
      for image_prompt in image_prompts:
        image_embed = clip_model.encode_image(self.preprocess(image_prompt))
        self.target_embeds.append(image_embed)

    self.target_embeds = torch.cat(self.target_embeds)

    self.n_cuts = n_cuts

  def forward(self, input):
    if self.n_cuts > 1:
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.clip_model_input_size)
        cutouts = []
        for _ in range(self.n_cuts):
            size = int(torch.rand([]) * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.clip_model_input_size))
        input = torch.cat(cutouts)

    input_embed = self.clip_model.encode_image(self.preprocess(input))  
    input_normed = torch.nn.functional.normalize(input_embed.unsqueeze(1), dim=-1)
    embed_normed = torch.nn.functional.normalize(self.target_embeds.unsqueeze(0), dim=-1)
    dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)

    return dists.mean()

class LossCLIP(Loss[LossCLIPCfg, LossCLIPCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)

        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
        self.clip_model = clip_model
        self.clip_model_input_size = 224
        self.preprocess = transforms.Compose([
            transforms.Resize(size=self.clip_model_input_size, max_size=None, antialias=None),
            transforms.CenterCrop(size=(self.clip_model_input_size, self.clip_model_input_size)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.clip_model
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.n_cuts = cfg.clip.n_cuts

    def _forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Rearrange and mask predicted and ground truth images
        B, S, C, H, W = prediction.color.shape
        rendered_imgs = prediction.color
        style_imgs = ((batch["context"]["style_image"] +1) / 2)

        rendered_imgs = rendered_imgs.view(B, S, C, H, W)
        style_imgs = style_imgs.view(B, 1, C, H, W)

        total_loss = 0.0
        for b in range(B):
            total_loss += self._clip_loss(rendered_imgs[b],style_imgs[b])
        total_loss /= B
        return total_loss* self.cfg.weight

    def _clip_loss(self, inputs: Tensor, target: Tensor) -> Float[Tensor, ""]:
        S, C, H, W = inputs.shape
        target = target.view(1, C, H, W)
        if self.cfg.cut_target:
            target = self.generate_cutouts(target)
        target_embeds = self.clip_model.encode_image(self.preprocess(target))
        target_normed = torch.nn.functional.normalize(target_embeds, dim=-1).unsqueeze(0)
        if self.n_cuts > 1:
            inputs = [self.generate_cutouts(inputs[b].unsqueeze(0)) for b in range(S)]
            inputs = torch.cat(inputs, dim=0)
        input_embed = self.clip_model.encode_image(self.preprocess(inputs))  
        input_normed = torch.nn.functional.normalize(input_embed.unsqueeze(1), dim=-1)
        # Compute the difference between normalized embeddings
        diff = input_normed.sub(target_normed)
        
        # Compute the L2 norm along dimension 2
        norm = diff.norm(dim=2)
        
        # Divide by 2
        half_norm = norm.div(2)
        
        # Apply arcsin
        arcsin_val = half_norm.arcsin()
        
        # Square the result
        squared = arcsin_val.pow(2)
        
        # Multiply by 2 to get final distances
        dists = squared.mul(2)
        loss =  dists.mean()
        return loss

    def generate_cutouts(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.clip_model_input_size)
        cutouts = []
        for _ in range(self.n_cuts):
            size = int(torch.rand([]) * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.clip_model_input_size))
        input = torch.cat(cutouts)
        return input
    

    

    def forward(self, prediction, batch, gaussians, depth_dict, global_step):
        return self._forward(prediction, batch, gaussians, depth_dict, global_step)


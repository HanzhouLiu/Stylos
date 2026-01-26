# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DPT style head with cross-attention for Stylos
# --------------------------------------------------------

from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from .head_modules import UnetExtractor, AppearanceTransformer, _init_weights
from .postprocess import postprocess

from einops import rearrange

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


# -------------------------
# Cross Attention Module
# https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
# -------------------------
class ChannelCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q from deep_feat, K/V from shallow_feat
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)

        # depthwise conv like MDTA
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, deep_feat, shallow_feat):
        B, C, H, W = deep_feat.shape

        # Q,K,V projection + DWConv
        q = self.q_dwconv(self.q_proj(deep_feat))
        kv = self.kv_dwconv(self.kv_proj(shallow_feat))
        k, v = kv.chunk(2, dim=1)

        # split heads: [B, head, C_per_head, HW]
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        # L2 normalize on channel-dim
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        # channel attention [B, head, C_per_head, C_per_head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # apply attention
        out = attn @ v  # [B, head, C_per_head, HW]

        # reshape back
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=H, w=W)

        return self.project_out(out) + deep_feat


# -------------------------
# Style Head
# -------------------------
class VGGT_DPT_Style_Head(DPTHead):
    def __init__(self, 
            dim_in: int,
            patch_size: int = 14,
            output_dim: int = 83,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            feature_only: bool = False,
            down_ratio: int = 1,
            resize_method: str = "deconv",
            geo_dim: int = 3,
    ):
        super().__init__(dim_in, patch_size, output_dim, activation,
                         conf_activation, features, out_channels,
                         intermediate_layer_idx, pos_embed,
                         feature_only, down_ratio, resize_method)
        
        head_features = 128

        # Shallow extractor for high-frequency cues
        self.shallow_extractor = nn.Sequential(
            nn.Conv2d(3, head_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features, head_features, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Cross attention fusion block
        # self.cross_attn = StyleCrossAttention(dim=head_features)
        self.cross_attn = ChannelCrossAttention(dim=head_features)

        self.geo_dim = geo_dim
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features, head_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features, output_dim, kernel_size=1, stride=1, padding=0),
        )
        
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def forward(self, encoder_tokens: List[torch.Tensor], depths, imgs,
                patch_start_idx: int = 5, image_size=None, conf=None, frames_chunk_size: int = 8):

        B, S, _, H, W = imgs.shape
        imgs = (imgs - self._resnet_mean) / self._resnet_std
    
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, imgs, patch_start_idx)

        all_preds = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_output = self._forward_impl(
                encoder_tokens, imgs, patch_start_idx,
                frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)
        return torch.cat(all_preds, dim=1)

    def _forward_impl(self, encoder_tokens: List[torch.Tensor], imgs,
                      patch_start_idx: int = 5,
                      frames_start_idx: int = None,
                      frames_end_idx: int = None):

        if frames_start_idx is not None and frames_end_idx is not None:
            imgs = imgs[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = imgs.shape

        if isinstance(self.patch_size, int):
            patch_h, patch_w = H // self.patch_size, W // self.patch_size
        else:
            patch_h, patch_w = H // self.patch_size[0], W // self.patch_size[1]

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][:, :, patch_start_idx:]
            
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()
            
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1

        # Deep features from encoder
        out = self.scratch_forward(out)  # [B*S, C, H', W']

        # Shallow features from input images (keep high freq, then resize)
        shallow = self.shallow_extractor(imgs.flatten(0,1))  
        # shallow = F.interpolate(shallow, size=out.shape[-2:], mode="bilinear", align_corners=True)

        # Upsample to full resolution
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        # Cross attention fusion
        out = self.cross_attn(out, shallow)

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out = self.scratch.output_conv2(out)
        out = out.view(B, S, *out.shape[1:])
        return out

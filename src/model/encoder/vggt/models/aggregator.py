# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from src.model.encoder.vggt.layers import PatchEmbed, VGG19PatchEmbed
from src.model.encoder.vggt.layers.block import Block, CrossBlock, CrossBlock2
from src.model.encoder.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from src.model.encoder.vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        self.use_checkpoint = True
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        if "frame" in aa_order:
            self.frame_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )
        if "global" in aa_order:
            self.global_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )
    
    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """
        
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        intermediate_layer_idx: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        patch_tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        tokens = patch_tokens

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        layer_idx = 0
        
        # Convert intermediate_layer_idx to a set for O(1) lookup
        if intermediate_layer_idx is not None:
            required_layers = set(intermediate_layer_idx)
            # Always include the last layer for camera_head
            required_layers.add(self.depth - 1)

        for _ in range(self.aa_block_num):
            intermediate_len = 0
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    intermediate_len = len(frame_intermediates)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    intermediate_len = len(global_intermediates)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if intermediate_layer_idx is not None:
                for i in range(intermediate_len):
                    current_layer = layer_idx + i
                    if current_layer in required_layers:
                        # concat frame and global intermediates, [B x S x P x 2C]
                        if self.aa_order == ["frame", "global"]:
                            concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                        elif self.aa_order == ["global"]:
                            concat_inter = global_intermediates[i]
                        elif self.aa_order == ["frame"]:
                            concat_inter = frame_intermediates[i]
                        else:
                            raise ValueError(f"Unknown aa_order: {self.aa_order}")
                        output_list.append(concat_inter)
                layer_idx += self.aa_block_size
            
            else:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    if self.aa_order == ["frame", "global"]:
                        concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    elif self.aa_order == ["global"]:
                        concat_inter = global_intermediates[i]
                    elif self.aa_order == ["frame"]:
                        concat_inter = frame_intermediates[i]
                    else:
                        raise ValueError(f"Unknown aa_order: {self.aa_order}")
                    output_list.append(concat_inter)
        
        del concat_inter
        if "frame" in self.aa_order:
            del frame_intermediates
        if "global" in self.aa_order:
            del global_intermediates
        return output_list, self.patch_start_idx, patch_tokens, pos

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


class StyleAggregator(nn.Module):
    """
    A style aggregator that processes images with alternating attention.
    It uses the Aggregator class to handle the attention mechanism.
    """

    def __init__(
        self,
        aggregator: Aggregator,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn="CrossBlock",
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        shared_patch_embed=False,
        encode_content=False,
        fuse_patch_tokens=True,
        expand_style_tokens=1,
    ):
        super().__init__()
        self.use_checkpoint = True
        self.aa_order= aa_order
        self.encode_content = encode_content
        self.shared_patch_embed = shared_patch_embed
        self.fuse_patch_tokens = fuse_patch_tokens
        self.expand_style_tokens = expand_style_tokens
        
        if block_fn=="CrossBlock":
            if "mix" in aa_order:
                global_block_fn = Block
                frame_block_fn = CrossBlock
            else:
                global_block_fn = CrossBlock
                frame_block_fn = CrossBlock
        elif block_fn=="CrossBlock2":
            if "mix" in aa_order:
                global_block_fn = Block
                frame_block_fn = CrossBlock2
            else:
                global_block_fn = CrossBlock2
                frame_block_fn = CrossBlock2
        else:
            raise ValueError(f"Unknown block_fn: {block_fn}. Use 'CrossBlock' or 'CrossBlock2'.")
        # self.patch_embed = aggregator.patch_embed
        # self.patch_embed.eval()  # Set to eval mode to avoid training updates
        # # disable patch_embed gradient updates
        # for param in self.patch_embed.parameters():
        #     param.requires_grad = False
        self.aggregator = aggregator
        if shared_patch_embed:
            self.patch_embed = aggregator.patch_embed
            self.rope = aggregator.rope
            self.position_getter = aggregator.position_getter
        else:
            self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)
        # Initialize rotary position embedding if frequency > 0
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        
        if "frame" in aa_order:
            self.frame_blocks = nn.ModuleList(
                [
                    frame_block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )
        if "global" in aa_order:
            self.global_blocks = nn.ModuleList(
                [
                    global_block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )
        if "mix" in aa_order:
            self.frame_blocks = nn.ModuleList(
                [
                    frame_block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )
            self.global_blocks = nn.ModuleList(
                [
                    global_block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.aggregator.rope,
                    )
                    for _ in range(depth)
                ]
            )
        self.depth = depth
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size


       # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.register_token, std=1e-6)
        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )        

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """
        
        if "conv" == patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        elif "vgg19" == patch_embed:
            self.patch_embed = VGG19PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        elif patch_embed in ["dinov2_vitl14_reg", "dinov2_vitb14_reg", "dinov2_vits14_reg", "dinov2_vitg2_reg"]:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
        else:
            raise ValueError(f"Unknown patch embed type: {patch_embed}")


    def forward(self, 
                style_images: torch.Tensor, 
                images: torch.Tensor, 
                image_patch_tokens: torch.Tensor, 
                image_pos: torch.Tensor,
                image_patch_start_idx: int,
                intermediate_layer_idx: Optional[List[int]]) -> List[torch.Tensor]:
        return self._forward(style_images, images, image_patch_tokens, image_pos, image_patch_start_idx, intermediate_layer_idx)

    def _forward(self, 
                style_images: torch.Tensor, 
                images: torch.Tensor, 
                image_patch_tokens: torch.Tensor, 
                image_pos: torch.Tensor,
                image_patch_start_idx: int,
                intermediate_layer_idx: Optional[List[int]]) -> List[torch.Tensor]:
        """
        Forward pass through the style aggregator.

        Args:
            style_images (torch.Tensor): Input style images with shape [B, 3, H, W].
            patch_start_idx (int): Starting index for the patch.

        Returns:
            Tuple[List[torch.Tensor], int]: Outputs from the aggregator and patch_start_idx.
        """
        B, S, C_in, H, W = images.shape
        if style_images.shape[1] != 1:
            raise ValueError(f"Expected 1 style image, got {S}")
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # Normalize images and reshape for patch embed
        style_images = (style_images - self._resnet_mean) / self._resnet_std
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        style_images = style_images.view(B*1 , C_in, H, W)
        patch_tokens = self.patch_embed(style_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
    
        if self.encode_content and not self.shared_patch_embed:
            if self.fuse_patch_tokens:
                patch_tokens += self.aggregator.patch_embed(style_images)["x_norm_patchtokens"]
            images = images.view(B*S, C_in, H, W)
            content_image_patch_tokens = self.patch_embed(images)
            if isinstance(content_image_patch_tokens, dict):
                content_image_patch_tokens = content_image_patch_tokens["x_norm_patchtokens"]
            if self.fuse_patch_tokens:
                image_patch_tokens[:, image_patch_start_idx:, :] += content_image_patch_tokens
            else:
                image_patch_tokens[:, image_patch_start_idx:, :] = content_image_patch_tokens


        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        register_token = self.register_token.expand(B, 1, *self.register_token.shape[2:])
        register_token = register_token.view(B*1,*self.register_token.shape[2:])  

        style_tokens = torch.cat([register_token, patch_tokens], dim=1)
        #image_patch_tokens[:, image_patch_start_idx:, :] = style_image_patch_tokens
        
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * 1, H // self.patch_size, W // self.patch_size, device=style_images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * 1, self.patch_start_idx, 2).to(style_images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        _, q_P, C = image_patch_tokens.shape
        _, kv_P, C = style_tokens.shape
        if self.expand_style_tokens>1 and q_P>self.expand_style_tokens:
            expand_size = q_P//self.expand_style_tokens
            style_tokens = style_tokens.unsqueeze(1).expand(B, expand_size, kv_P, C).contiguous()
            style_tokens = style_tokens.view(B, expand_size*kv_P, C)
            print(f"Expand style tokens from {kv_P} to {expand_size*kv_P}")
            if pos is not None:
                pos = pos.unsqueeze(1).expand(B, expand_size, kv_P, 2).contiguous()
                pos = pos.view(B, expand_size*kv_P, 2)
            kv_P = expand_size*kv_P
        # Convert intermediate_layer_idx to a set for O(1) lookup
        if intermediate_layer_idx is not None:
            required_layers = set(intermediate_layer_idx)
            # Always include the last layer for camera_head
            required_layers.add(self.depth - 1)
        if "mix" in self.aa_order:
            # for gframe, we need to process frame and global attention together
            output_list = self.mixedatt_forward(image_patch_tokens, image_pos, intermediate_layer_idx, B, S, C, style_tokens, pos, q_P, kv_P, required_layers)
        else:
            output_list = self.allcrossatt_forward(image_patch_tokens, image_pos, intermediate_layer_idx, B, S, C, style_tokens, pos, q_P, kv_P, required_layers)
        return output_list

    def allcrossatt_forward(self, image_patch_tokens, image_pos, intermediate_layer_idx, B, S, C, style_tokens, pos, q_P, kv_P, required_layers):
        frame_idx = 0
        global_idx = 0
        style_idx = 0
        output_list = []
        layer_idx = 0
        q_tokens = image_patch_tokens
        kv_tokens = style_tokens
        q_pos = image_pos
        kv_pos = pos
        for _ in range(self.aa_block_num):
                intermediate_len = 0
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        q_tokens, kv_tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            q_tokens, kv_tokens, B, S, q_P, kv_P, C, frame_idx, self.frame_blocks, q_pos=q_pos, kv_pos=kv_pos
                        )
                        intermediate_len = len(frame_intermediates)
                    elif attn_type == "global":
                        q_tokens, kv_tokens, global_idx, global_intermediates = self._process_global_attention(
                            q_tokens, kv_tokens, B, S, q_P, kv_P, C, global_idx, self.global_blocks, q_pos=q_pos, kv_pos=kv_pos
                        )
                        intermediate_len = len(global_intermediates)
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                if intermediate_layer_idx is not None:
                    for i in range(intermediate_len):
                        current_layer = layer_idx + i
                        if current_layer in required_layers:
                            # concat frame and global intermediates, [B x S x P x 2C]
                            if self.aa_order == ["frame", "global"] or self.aa_order == ["global", "frame"]:
                            # Note: we allow both orders for flexibility
                                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                            elif self.aa_order == ["global"]:
                                concat_inter = global_intermediates[i]
                            elif self.aa_order == ["frame"]:
                                concat_inter = frame_intermediates[i]
                            else:
                                raise ValueError(f"Unknown aa_order: {self.aa_order}")                            
                            output_list.append(concat_inter)
                    layer_idx += self.aa_block_size
                
                else:
                    for i in range(intermediate_len):
                        # concat frame and global intermediates, [B x S x P x 2C]
                        if self.aa_order == ["frame", "global"] or self.aa_order == ["global", "frame"]:
                            concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                        elif self.aa_order == ["global"]:
                            concat_inter = global_intermediates[i]
                        elif self.aa_order == ["frame"]:
                            concat_inter = frame_intermediates[i]
                        else:
                            raise ValueError(f"Unknown aa_order: {self.aa_order}")
                        output_list.append(concat_inter)
        del concat_inter
        if "frame" in self.aa_order:
            del frame_intermediates
        if "global" in self.aa_order:
            del global_intermediates
        return output_list


    def mixedatt_forward(self, image_patch_tokens, image_pos, intermediate_layer_idx, B, S, C, style_tokens, pos, q_P, kv_P, required_layers):
        frame_idx = 0
        global_idx = 0
        style_idx = 0
        output_list = []
        layer_idx = 0
        q_tokens = image_patch_tokens
        kv_tokens = style_tokens
        q_pos = image_pos
        kv_pos = pos

        for _ in range(self.aa_block_num):
                intermediate_len = 0
                q_tokens, global_idx, global_intermediates = self._process_global_self_attention(
                    q_tokens, B, S, q_P, C, global_idx, pos=q_pos
                )
                intermediate_len = len(global_intermediates)
                q_tokens, kv_tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                    q_tokens, kv_tokens, B, S, q_P, kv_P, C, frame_idx, self.frame_blocks, q_pos=q_pos, kv_pos=kv_pos
                )
                intermediate_len = len(frame_intermediates)
                if intermediate_layer_idx is not None:
                    for i in range(intermediate_len):
                        current_layer = layer_idx + i
                        if current_layer in required_layers:
                            concat_inter = torch.cat([global_intermediates[i], frame_intermediates[i]], dim=-1)
                            output_list.append(concat_inter)
                    layer_idx += self.aa_block_size
                
                else:
                    raise NotImplementedError("intermediate_layer_idx must be provided for gframe attention")

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list
    
    def _process_frame_attention(self, q_tokens, kv_tokens, B, S,  q_P, kv_P, C, frame_idx, frame_blocks, q_pos, kv_pos):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if q_tokens.shape != (B*S, q_P, C):
            q_tokens = q_tokens.view(B, S, q_P, C).view(B * S, q_P, C)
        if kv_tokens.shape != (B * S, kv_P, C):
            kv_tokens = kv_tokens.view(B, 1, kv_P, C).expand(B, S, kv_P, C).contiguous()
            kv_tokens = kv_tokens.view(B * S, kv_P, C)
            # remove expand when return kv_tokens

        if q_pos is not None and q_pos.shape != (B * S, q_P, 2):
            q_pos = q_pos.view(B, S, q_P, 2).view(B * S, q_P, 2)

        if kv_pos is not None and kv_pos.shape != (B * S, kv_P, 2):
            kv_pos = kv_pos.view(B, 1, kv_P, 2).expand(B, S, kv_P, 2).contiguous()
            kv_pos = kv_pos.view(B * S, kv_P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                q_tokens, kv_tokens = torch.utils.checkpoint.checkpoint(
                    frame_blocks[frame_idx],
                    q_tokens,
                    kv_tokens,
                    q_pos=q_pos,
                    kv_pos=kv_pos,
                    use_reentrant=False,
                )
            else:
                q_tokens, kv_tokens  = frame_blocks[frame_idx](q_tokens, kv_tokens, q_pos=q_pos, kv_pos=kv_pos)
            frame_idx += 1
            intermediates.append(q_tokens.view(B, S, q_P, C))
        kv_tokens = kv_tokens.view(B, S, kv_P, C)
        return q_tokens, kv_tokens[:, 0], frame_idx, intermediates

    def _process_global_attention(self, q_tokens, kv_tokens, B, S, q_P, kv_P, C, global_idx, global_blocks, q_pos=None, kv_pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        # If needed, reshape tokens or positions:
        if q_tokens.shape != (B , S * q_P, C):
            q_tokens = q_tokens.view(B, S, q_P, C).view(B, S*q_P, C)
        if kv_tokens.shape != (B, 1*kv_P, C):
            kv_tokens = kv_tokens.view(B, 1, kv_P, C).view(B, 1*kv_P, C)

        if q_pos is not None and q_pos.shape != (B, S * q_P, 2):
            q_pos = q_pos.view(B, S, q_P, 2).view(B, S * q_P, 2)

        if kv_pos is not None and kv_pos.shape != (B, 1*kv_P, 2):
            kv_pos = kv_pos.view(B, 1, kv_P, 2).view(B, 1*kv_P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                q_tokens, kv_tokens = torch.utils.checkpoint.checkpoint(
                    global_blocks[global_idx],
                    q_tokens,
                    kv_tokens,
                    q_pos=q_pos,
                    kv_pos=kv_pos,
                    use_reentrant=False,
                )
            else:
                q_tokens, kv_tokens = global_blocks[global_idx](q_tokens, kv_tokens, q_pos=q_pos, kv_pos=kv_pos)
            global_idx += 1
            intermediates.append(q_tokens.view(B, S, q_P, C))

        return q_tokens, kv_tokens, global_idx, intermediates

    def _process_global_self_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates
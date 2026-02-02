import os
from copy import deepcopy
import time
from typing import Optional
from einops import rearrange
import huggingface_hub
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from safetensors.torch import load_file, save_file

from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.model.encoder.anysplat import EncoderAnySplat, EncoderAnySplatCfg, OpacityMappingCfg


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

class AnySplat(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_cfg: EncoderAnySplatCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):  
        super(AnySplat, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)
        # self.load_pretrained_backbone("/workspace/Code/StyleAnySplat/checkpoints/model.safetensors",freeze=True)

    def load_pretrained_backbone(self, checkpoint_path, freeze: bool = True):
        """Load pretrained weights from a given path."""
        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path)
        elif checkpoint_path.endswith('.pth') or checkpoint_path.endswith('.pt'):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)
        # copy params of encoder.gaussian_param_head to encoder.style_gaussian_param_head
        # self.encoder.style_gaussian_param_head.load_state_dict(self.encoder.gaussian_param_head.state_dict(), strict=False, assign=True)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for module in [self.encoder.style_aggregator, self.encoder.style_gaussian_param_head]:
                for param in module.parameters():
                    param.requires_grad = True
            if hasattr(self.encoder.style_aggregator, 'style_aggregator') and hasattr(self.encoder.style_aggregator, 'aggregator'):
                for param in self.encoder.style_aggregator.aggregator.parameters():
                    param.requires_grad = False

        num_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        print(f"Encoder has {num_params} trainable parameters out of {encoder_total}")
        # print decoder
        num_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f"Decoder has {num_params} trainable parameters.")

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        """Convert nested dictionary config to dataclass instance
        
        Args:
            cfg_dict: Configuration dictionary or already converted object
            target_class: Target dataclass type to convert to
            
        Returns:
            Instance of target_class
        """
        if isinstance(cfg_dict, dict):
            # Convert dict to dataclass
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            # Already converted, return as is
            return cfg_dict
        elif cfg_dict is None:
            # Handle None case
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        """Convert nested configurations recursively using a conversion map
        
        Args:
            cfg_obj: Configuration object to convert
            conversion_map: Dict mapping field names to their target classes
                           e.g., {'gaussian_adapter': GaussianAdapterCfg}
        
        Returns:
            Converted configuration object
        """
        if not hasattr(cfg_obj, '__dict__'):
            return cfg_obj
            
        cfg_dict = cfg_obj.__dict__.copy()
        
        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(
                    cfg_dict[field_name], 
                    target_class
                )
        
        # Return new instance of the same type
        return type(cfg_obj)(**cfg_dict)

    def convert_encoder_config(self, encoder_cfg: EncoderAnySplatCfg) -> EncoderAnySplatCfg:
        """Convert all nested configurations in encoder_cfg"""
        conversion_map = {
            'gaussian_adapter': GaussianAdapterCfg,
            'opacity_mapping': OpacityMappingCfg,
        }
        
        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderAnySplatCfg):
        # Convert nested configurations using the helper method
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderAnySplat(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)
    
    @torch.no_grad()
    def inference(self,
        context_image: torch.Tensor,
        style_image: torch.Tensor,
    ):
        self.encoder.distill = False
        encoder_output = self.encoder(context_image, style_image, global_step=0, visualization_dump=None)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        return gaussians, pred_context_pose
    
    def forward(self, 
        context_image: torch.Tensor,
        style_image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        near: float = 0.01,
        far: float = 100.0,
        gt_extrinsics: Optional[torch.Tensor] = None,
        gt_intrinsics: Optional[torch.Tensor] = None,
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device
        encoder_output = self.encoder(context_image, style_image, global_step, visualization_dump=visualization_dump)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        if gt_extrinsics is not None and gt_intrinsics is not None:
            output = self.decoder.forward(
                gaussians,
                gt_extrinsics,
                gt_intrinsics,
                torch.ones(1, v, device=device) * near,
                torch.ones(1, v, device=device) * far,
                (h, w),
                "depth",
            )
        else:
            output = self.decoder.forward(
                gaussians,
                pred_context_pose['extrinsic'],
                pred_context_pose["intrinsic"],
                torch.ones(1, v, device=device) * near,
                torch.ones(1, v, device=device) * far,
                (h, w),
                "depth",
            )
        return encoder_output, output
    

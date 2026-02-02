from dataclasses import dataclass
from functools import partial
from typing import List, Optional

from jaxtyping import Float
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch_scatter import scatter_add, scatter_max

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss, T_cfg, T_wrapper, ScaledMSELoss, VGG19Features
from .sqrtm import sqrtm_ns_lyap

import math

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def gram_matrix(feats: torch.Tensor):
    """
    feats: [B, C, N] or [B, C, H, W]
    return: [B, C, C]
    """
    if feats.dim() == 4:  # [B, C, H, W]
        B, C, H, W = feats.shape
        F = feats.view(B, C, -1)  # [B, C, N]
    elif feats.dim() == 3:  # [B, C, N]
        B, C, N = feats.shape
        F = feats
    else:
        raise ValueError("Unsupported feature shape")

    N = F.size(-1)
    G = torch.bmm(F, F.transpose(1, 2))  # [B, C, C]
    return G / (C * N)  # 归一化


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4) or (len(size) == 5) or (len(size) == 3), "Input feature map must be 4D or 5D"
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        # self.enc_1 = nn.Sequential(*enc_layers[:3])  # input -> relu1_1
        # self.enc_2 = nn.Sequential(*enc_layers[3:10])  # relu1_1 -> relu2_1
        # self.enc_3 = nn.Sequential(*enc_layers[10:17])  # relu2_1 -> relu3_1
        # self.enc_4 = nn.Sequential(*enc_layers[17:30])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(5):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size()) or (input.dim() == 5 and input.size()[:2] == target.size()[:2])
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, content_images, style_images, stylized_images):
        if content_images.dim() == 3:
            content_images = content_images.unsqueeze(0)
        if style_images.dim() == 3:
            style_images = style_images.unsqueeze(0)
        if stylized_images.dim() == 3:
            stylized_images = stylized_images.unsqueeze(0)
        style_feats = self.encode_with_intermediate(style_images)
        content_feat = self.encode(content_images)
        stylized_feats = self.encode_with_intermediate(stylized_images)

        loss_c = self.calc_content_loss(stylized_feats[-1], content_feat)
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
        return loss_c, loss_s


@dataclass
class LossAdaINStyleCfg:
    weight: float = 1.0
    style_weight: float = 10.0
    content_weight: float = 1.0
    style_feat_weights: Optional[List[int]] = None 
    content_feat_weights: Optional[List[int]] = None
    mse_weight: float = 0.0
    style_loss_type: str = "image"  # "scene", "image" , "3d"
    use_checkpoint: bool = False
    voxel_size: float = 0.002


@dataclass
class LossAdaINStyleCfgWrapper:
    adain_style: LossAdaINStyleCfg


class LossAdaINStyle(Loss[LossAdaINStyleCfg, LossAdaINStyleCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)

        vgg.load_state_dict(torch.load("checkpoints/vgg_normalised.pth"))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.net = Net(vgg)
        self.net.eval()
        # self.net.requires_grad_(False)
        if self.cfg.style_feat_weights is None:
            self.cfg.style_feat_weights = [256, 64, 16, 4, 1]
        if self.cfg.content_feat_weights is None:
            self.cfg.content_feat_weights = [4,1]
        self.style_feat_weights = np.array(self.cfg.style_feat_weights)
        self.style_feat_weights = self.style_feat_weights / np.sum(self.style_feat_weights)
        self.content_feat_weights = np.array(self.cfg.content_feat_weights)
        self.content_feat_weights = self.content_feat_weights / np.sum(self.content_feat_weights)

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
        content_imgs = ((batch["context"]["image"] +1) / 2)
        rendered_imgs = prediction.color
        style_imgs = ((batch["context"]["style_image"] +1) / 2)

        content_imgs = content_imgs.view(B * S, C, H, W)
        rendered_imgs = rendered_imgs.view(B * S, C, H, W)
        style_imgs = style_imgs.view(B * 1, C, H, W)
        content_imgs = self.normalize(content_imgs)
        rendered_imgs = self.normalize(rendered_imgs)
        style_imgs = self.normalize(style_imgs)

        content_imgs = content_imgs.view(B, S, C, H, W)
        rendered_imgs = rendered_imgs.view(B, S, C, H, W)
        style_imgs = style_imgs.view(B, 1, C, H, W)
        pts_all = prediction.pts_all
        conf = prediction.conf

        # Style Loss (mean/variance differences)
        if self.cfg.style_loss_type == "scene":
            # Use scene-level style loss
            style_loss, content_loss, mse_loss = self.calculate_scene_losses(content_imgs, rendered_imgs, style_imgs)
        elif self.cfg.style_loss_type == "3d":
            style_loss, content_loss, mse_loss = self.calculate_3d_losses(content_imgs, rendered_imgs, style_imgs, pts_all, conf, depth_dict['conf_valid_mask'])
        else:
            style_loss, content_loss, mse_loss = self.calculate_image_losses(content_imgs, rendered_imgs, style_imgs)

        total_loss = self.cfg.content_weight * content_loss.mean() / (B * S) + \
            self.cfg.style_weight * style_loss.mean() / (B * S)
        if self.cfg.mse_weight > 0:
            total_loss += mse_loss.mean() / (B * S)

        return total_loss * self.cfg.weight
    
    def calculate_scene_losses(self, content_imgs, rendered_imgs, style_imgs):
        B, S, C, H, W = content_imgs.shape

        style_imgs = style_imgs.view(B, C, H, W)
        style_feats = self.net.encode_with_intermediate(style_imgs)

        content_imgs = content_imgs.view(B * S, C, H, W)
        content_feats = self.net.encode_with_intermediate(content_imgs)

        rendered_imgs = rendered_imgs.view(B * S, C, H, W)
        rendered_feats = self.net.encode_with_intermediate(rendered_imgs)
        # calculate content loss for the last two layers
        content_loss = self.net.calc_content_loss(rendered_feats[-1], content_feats[-1]) * self.content_feat_weights[0]
        content_loss += self.net.calc_content_loss(rendered_feats[-2], content_feats[-2]) * self.content_feat_weights[1]

        mse_loss = 0.0
        if self.cfg.mse_weight > 0:
            # Calculate MSE loss
            mse_loss += F.mse_loss(rendered_imgs, content_imgs) * self.cfg.mse_weight

        # calculate style loss for the first layer
        style_loss = 0.0
        for i in range(5):
            N, fc, fh, fw = rendered_feats[i].shape
            style_loss += self.net.calc_style_loss(rendered_feats[i].view(B, S, fc, fh, fw).transpose(1, 2).contiguous(), style_feats[i]) * self.style_feat_weights[i]
        return style_loss, content_loss, mse_loss

    def voxelization_with_fusion(self, img_feat, pts3d, voxel_size, conf=None, valid_mask=None, weighting_mode = "softmax"):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        if valid_mask is None:
            pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)
            # Flatten confidence scores and features
            conf_flat = conf.flatten()  # [B*V*N]
            anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...] 
        else:
            valid_mask_flat = valid_mask.flatten(0, 2)  # [B*V*N]
            pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)[valid_mask_flat]
            conf_flat = conf.flatten()[valid_mask_flat]
            anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)[valid_mask_flat]

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        if weighting_mode == "uniform":
            weights = (1.0 / counts[inverse_indices])  # [B*V*N, 1]
        elif weighting_mode == "l1":
            # Clamp to avoid negative weights
            conf_pos = torch.clamp(conf_flat, min=0.0)  # [N]

            # Per-voxel sum
            sum_conf = scatter_add(conf_pos, inverse_indices, dim=0)  # [num_vox]

            # Normalize (L1 normalization)
            weights = conf_pos / (sum_conf[inverse_indices] + 1e-12)  # [N]

            # Fallback to uniform if sum is zero
            voxel_counts = scatter_add(torch.ones_like(conf_flat), inverse_indices, dim=0)
            counts_per_point = voxel_counts[inverse_indices]
            uniform_w = 1.0 / torch.clamp(counts_per_point, min=1.0)
            weights = torch.where(sum_conf[inverse_indices] > 0, weights, uniform_w)
        elif weighting_mode == "softmax":
            # --- Per-voxel normalized weights (stable softmax) ---
            # inverse_indices: [N] maps each point to its voxel id (0..num_unique_voxels-1)
            # conf_flat: [N] confidences per point (can be any real values)

            # 1) subtract per-voxel max for numerical stability
            conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)           # [num_vox]
            stable = conf_flat - conf_voxel_max[inverse_indices]                          # [N]

            # 2) optional temperature to control sharpness (tau=1 keeps behavior)
            tau = 1.0
            stable = stable / tau

            # 3) exponentiate and sum per voxel
            conf_exp = torch.exp(stable)                                                  # [N]
            sum_exp = scatter_add(conf_exp, inverse_indices, dim=0)                       # [num_vox]

            # 4) avoid divide-by-zero; if a voxel had all -inf or NaNs, fall back to uniform
            # Build per-voxel counts
            voxel_counts = scatter_add(torch.ones_like(conf_flat), inverse_indices, dim=0)  # [num_vox]
            safe_den = sum_exp + 1e-12

            # Compute weights; they sum to 1 per voxel
            weights = conf_exp / safe_den[inverse_indices]                                # [N]

            # 5) Uniform fallback where sum_exp==0 (degenerate): each point gets 1/count
            degenerate = (sum_exp <= 0) | ~torch.isfinite(sum_exp)                         # [num_vox]
            if degenerate.any():
                # map voxel_counts to points
                counts_per_point = voxel_counts[inverse_indices]
                uniform_w = 1.0 / torch.clamp(counts_per_point, min=1.0)
                weights = torch.where(degenerate[inverse_indices], uniform_w, weights)
        else:
            raise ValueError(f"Unknown weighting mode: {weighting_mode}")

        # 6) Apply weights to features (no squeeze!)
        # anchor_feats_flat: [N, C]
        weighted_feats = anchor_feats_flat * weights.unsqueeze(-1)                    # [N, C]

        # 7) Aggregate per voxel
        voxel_feats = scatter_add(weighted_feats, inverse_indices, dim=0)             # [num_vox, C]
        return voxel_feats

    def calculate_3d_losses(self, content_imgs, rendered_imgs, style_imgs, pts_all, conf, conf_valid_mask):
        """
        pts_all [B,V,H,W,3]
        conf [B,V,H,W]
        conf_valid_mask [B,V,H,W]   
        """
        assert pts_all is not None, "pts_all is required for 3d style loss, set pass_pts_all to True"
        B, S, C, H, W = content_imgs.shape

        style_imgs = style_imgs.view(B, C, H, W)
        style_feats = self.net.encode_with_intermediate(style_imgs)

        content_imgs = content_imgs.view(B * S, C, H, W)
        content_feats = self.net.encode_with_intermediate(content_imgs)

        rendered_imgs = rendered_imgs.view(B * S, C, H, W)
        rendered_feats = self.net.encode_with_intermediate(rendered_imgs)
        # calculate content loss for the last two layers
        content_loss = self.net.calc_content_loss(rendered_feats[-1], content_feats[-1]) * self.content_feat_weights[0]
        content_loss += self.net.calc_content_loss(rendered_feats[-2], content_feats[-2]) * self.content_feat_weights[1]

        mse_loss = 0.0
        if self.cfg.mse_weight > 0:
            # Calculate MSE loss
            mse_loss += F.mse_loss(rendered_imgs, content_imgs) * self.cfg.mse_weight

        # calculate style loss for the first layer
        feat_size = []
        conf = conf.view(B*S, 1, H, W)
        conf_valid_mask = conf_valid_mask.view(B*S,1, H, W)
        pts_all = pts_all.view(B*S, H, W, 3).permute(0, 3, 1, 2).contiguous()
        style_loss = 0.0
        for i in range(5):
            N, fc, fh, fw = rendered_feats[i].shape
            rendered_feats[i] = rendered_feats[i].view(B,S, fc, fh, fw)
            if fh != H or fw != W:
                resize_conf = F.interpolate(conf, size=(fh, fw), mode='bilinear', align_corners=False) # (B*S, 1, fh, fw)
                resize_conf = resize_conf.view(B, S, fh, fw)
                resize_conf_valid_mask = F.interpolate(conf_valid_mask.float(), size=(fh, fw), mode='nearest-exact') # (B*S, 1, fh, fw)
                resize_conf_valid_mask = resize_conf_valid_mask.view(B, S, fh, fw)
                resize_conf_valid_mask = resize_conf_valid_mask.bool()
                resize_pts_all = F.interpolate(pts_all, size=(fh, fw), mode='nearest-exact') # (B*S, 3, fh, fw)
                resize_pts_all = resize_pts_all.view(B, S, 3, fh, fw)
            else:
                resize_conf = conf.view(B, S, fh, fw)
                resize_conf_valid_mask = conf_valid_mask.view(B, S, fh, fw)
                resize_pts_all = pts_all.view(B, S, 3, fh, fw)
            #scale = (H*W) / (fh*fw)
            scale = math.sqrt((H*W)/(fh*fw))
            for b_i in range(B):
                neural_feats = self.voxelization_with_fusion(
                    rendered_feats[i][b_i].view(S, fc, fh, fw),
                    resize_pts_all[b_i].contiguous(),
                    self.cfg.voxel_size*scale,
                    conf=resize_conf[b_i],
                    valid_mask=resize_conf_valid_mask[b_i],
                )
                rendered_mean, rendered_std = calc_mean_std(neural_feats.permute(1,0).unsqueeze(0))
                style_mean, style_std = calc_mean_std(style_feats[i][b_i].unsqueeze(0))
                tmp_loss= self.net.mse_loss(rendered_mean, style_mean) + self.net.mse_loss(rendered_std, style_std)
                style_loss += tmp_loss * self.style_feat_weights[i]
        return style_loss, content_loss, mse_loss

    def calculate_image_losses(self, content_imgs, rendered_imgs, style_imgs):
        B, S, C, H, W = content_imgs.shape
        style_loss = 0.0
        content_loss = 0.0
        mse_loss = 0.0
        for i in range(B):
            style_image = style_imgs[i].view(1, C, H, W)
            style_feats = self.net.encode_with_intermediate(style_image)
            for j in range(S):
                content_image = content_imgs[i, j].view(1, C, H, W)
                rendered_image = rendered_imgs[i, j].view(1, C, H, W)
                if self.cfg.mse_weight > 0:
                    # Calculate MSE loss
                    mse_loss += F.mse_loss(rendered_image, content_image) * self.cfg.mse_weight
                # Calculate content loss
                content_feats = self.net.encode_with_intermediate(content_image)
                stylized_feats = self.net.encode_with_intermediate(rendered_image)
                for k in range(3, 5):
                    content_loss += self.net.calc_content_loss(stylized_feats[k], content_feats[k])

                # Calculate style loss
                stylized_feats = self.net.encode_with_intermediate(rendered_image)
                style_loss += self.net.calc_style_loss(stylized_feats[0], style_feats[0])
                for k in range(1, 5):
                    style_loss += self.net.calc_style_loss(stylized_feats[k], style_feats[k].detach())
        return style_loss,content_loss,mse_loss
    

    def forward(self, prediction, batch, gaussians, depth_dict, global_step):
        # if self.cfg.use_checkpoint:
        #     return torch.utils.checkpoint.checkpoint(
        #         self._forward, prediction, batch, gaussians, depth_dict, global_step
        #     )
        # else:
        return self._forward(prediction, batch, gaussians, depth_dict, global_step)

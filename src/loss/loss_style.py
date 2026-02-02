from dataclasses import dataclass
from functools import partial
from typing import List

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss, T_cfg, T_wrapper, ScaledMSELoss, VGG19Features
from .sqrtm import sqrtm_ns_lyap


class StyleLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss = ScaledMSELoss(eps=eps)

    @staticmethod
    def get_target(target):
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. (2015) and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input, target):
        return self.loss(self.get_target(input), self.get_target(target).detach())


def eye_like(x):
    return torch.eye(x.shape[-2], x.shape[-1], dtype=x.dtype, device=x.device).expand_as(x)


class StyleLossW2(nn.Module):
    """Wasserstein-2 style loss."""

    def __init__(self, eps=1e-4):
        super().__init__()

        self.sqrtm = partial(sqrtm_ns_lyap, num_iters=12)
        self.register_buffer('eps', torch.tensor(eps))
    @staticmethod
    def get_target(target):
        """Compute the mean and second raw moment of the target activations.
        Unlike the covariance matrix, these are valid to combine linearly."""
        mean = target.mean([-2, -1])
        srm = torch.einsum('...chw,...dhw->...cd', target, target) / (target.shape[-2] * target.shape[-1])
        return mean, srm

    @staticmethod
    def srm_to_cov(mean, srm):
        """Compute the covariance matrix from the mean and second raw moment."""
        return srm - torch.einsum('...c,...d->...cd', mean, mean)

    def forward(self, input, target):
        mean, srm = self.get_target(input)
        target_mean, target_srm = self.get_target(target)
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * self.eps
        target_cov = self.srm_to_cov(target_mean, target_srm) + eye_like(target_srm) * self.eps
        target_cov_sqrt = self.sqrtm(target_cov)
        mean_diff = torch.mean((mean - target_mean) ** 2)
        sqrt_term = self.sqrtm(target_cov_sqrt @ cov @ target_cov_sqrt)
        cov_diff = torch.diagonal(target_cov + cov - 2 * sqrt_term, dim1=-2, dim2=-1).mean()
        return mean_diff + cov_diff
    

class AdaINLoss(nn.Module):
    """Wasserstein-2 style loss."""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.sqrtm = partial(sqrtm_ns_lyap, num_iters=12)
        self.register_buffer('eps', torch.tensor(eps))
        self.mse_loss = nn.MSELoss()
    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    def forward(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input = input.unsqueeze(0) if input.dim() == 3 else input
        target = target.unsqueeze(0) if target.dim() == 3 else target
        input_mean, input_std = self.calc_mean_std(input, eps=self.eps)
        target_mean, target_std = self.calc_mean_std(target, eps=self.eps)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

class NNFM_Loss_MemoryEfficient(nn.Module):
    """
    Implements a memory-efficient version of the Nearest-Neighbor Feature Match (NNFM) loss.

    To avoid OOM errors with large feature maps, this version processes the
    feature vectors from the reconstructed image in chunks, avoiding the
    creation of a massive pairwise distance matrix.
    """
    def __init__(self, chunk_size: int = 2048):
        """
        Args:
            chunk_size (int): The number of feature vectors to process at a time.
                              Reduce this value if you encounter OOM errors.
        """
        super(NNFM_Loss_MemoryEfficient, self).__init__()
        self.chunk_size = chunk_size

    def forward(self, F_r: torch.Tensor, F_s: torch.Tensor) -> torch.Tensor:
        """
        Calculates the NNFM loss in a memory-efficient manner.

        Args:
            F_r (torch.Tensor): Feature map of the reconstructed/generated image.
                                Shape: (B, C, H_r, W_r)
            F_s (torch.Tensor): Feature map of the style image.
                                Shape: (B, C, H_s, W_s)

        Returns:
            torch.Tensor: The calculated NNFM loss, a scalar tensor.
        """
        assert (F_r.size() == F_s.size())
        assert (F_s.requires_grad is False)
        F_r = F_r.unsqueeze(0) if F_r.dim() == 3 else F_r
        F_s = F_s.unsqueeze(0) if F_s.dim() == 3 else F_s
        assert F_r.shape[1] == F_s.shape[1], "Feature maps must have the same number of channels."

        # Get dimensions
        B, C, H_r, W_r = F_r.shape
        _, _, H_s, W_s = F_s.shape
        
        N_r = H_r * W_r
        N_s = H_s * W_s

        # Reshape and normalize the style feature map once
        F_s_flat = F_s.view(B, C, N_s)
        F_s_norm = F.normalize(F_s_flat, p=2, dim=1)

        # Reshape the reconstructed feature map
        F_r_flat = F_r.view(B, C, N_r)
        
        # A list to store the minimum distances for each chunk
        all_min_dists = []
        
        # Process F_r in chunks to save memory
        for i in range(0, N_r, self.chunk_size):
            # Get a chunk of feature vectors from F_r
            F_r_chunk = F_r_flat[:, :, i : i + self.chunk_size]
            
            # Normalize the chunk
            F_r_chunk_norm = F.normalize(F_r_chunk, p=2, dim=1)

            # Calculate cosine similarity for the chunk against all of F_s
            # (B, C, N_s)T @ (B, C, chunk) -> (B, N_s, chunk) -> (B, chunk, N_s)
            sim_chunk = torch.bmm(F_r_chunk_norm.transpose(1, 2), F_s_norm)

            # Calculate cosine distance for the chunk
            dist_chunk = 1 - sim_chunk
            
            # Find the minimum distance for each vector in the chunk
            min_dists_chunk, _ = torch.min(dist_chunk, dim=2)
            all_min_dists.append(min_dists_chunk)

        # Concatenate the results from all chunks
        min_dists = torch.cat(all_min_dists, dim=1) # Shape: (B, N_r)

        # Final loss is the mean of all minimum distances
        loss = torch.mean(min_dists)
        
        return loss

@dataclass
class LossStyleCfg:
    weight: float = 1.0
    use_checkpoint: bool = False
    loss_type: str = 'w2'  # 'mse' or 'w2'

@dataclass
class LossStyleCfgWrapper:
    style: LossStyleCfg


class LossStyle(Loss[LossStyleCfg, LossStyleCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)
        self.layers = [1, 6, 11, 20, 29]  # Default to using the first four layers of VGG19
        self.style_weights = [1,1,1,1,1]#[256, 64, 16, 4, 1]
        self.vgg = VGG19Features(layers=self.layers )

        weight_sum = sum(abs(w) for w in self.style_weights)
        self.style_weights = [w / weight_sum for w in self.style_weights]
        if self.cfg.loss_type == 'mse':
            self.loss_fn = StyleLoss()
        elif self.cfg.loss_type == 'w2':
            self.loss_fn = StyleLossW2()
        elif self.cfg.loss_type == 'adain':
            self.loss_fn = AdaINLoss()
        elif self.cfg.loss_type == 'nnfm':
            self.loss_fn = NNFM_Loss_MemoryEfficient(chunk_size=1024)
        else:
            raise ValueError(f"Unsupported loss type: {self.cfg.loss_type}")
            

    # def forward(
    #     self,
    #     prediction: DecoderOutput,
    #     batch: BatchedExample,
    #     gaussians: Gaussians,
    #     depth_dict: dict,
    #     global_step: int,
    # ) -> Float[Tensor, ""]:

    #     # Rearrange and mask predicted and ground truth images
    #     B, S, C, H, W = prediction.color.shape
    #     rendered_img = prediction.color.view(B * S, C, H, W)
    #     style_img = ((batch["context"]["style_image"] + 1) / 2).view(B * 1, C, H, W)
    #     # Extract features
    #     rendered_feats = self.vgg(rendered_img)
    #     style_feats    = self.vgg(style_img)
        
    #     # Style Loss (mean/variance differences)
    #     style_loss = 0.0
    #     for layer in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']:
    #         mean_r, var_r = self.compute_mean_variance(rendered_feats[layer])
    #         _,C_s, fh, fw = mean_r.shape
    #         mean_r = mean_r.view(B, S, C_s, fh, fw)
    #         var_r = var_r.view(B, S, C_s, fh, fw)
    #         mean_s, var_s = self.compute_mean_variance(style_feats[layer])
    #         mean_s = mean_s.view(B, 1, C_s, fh, fw)
    #         var_s = var_s.view(B, 1, C_s, fh, fw)
    #         mean_s = mean_s.expand_as(mean_r)
    #         var_s = var_s.expand_as(var_r)

    #         style_loss += F.mse_loss(mean_r, mean_s) + F.mse_loss(var_r, var_s)

    #     return style_loss
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
        rendered_img = prediction.color.view(B * S, C, H, W)
        style_img = ((batch["context"]["style_image"] + 1) / 2).view(B * 1, C, H, W)
        # Extract features
        rendered_feats = self.vgg(rendered_img)
        style_feats    = self.vgg(style_img)

        # Style Loss (mean/variance differences)
        style_loss = 0.0
        for layer,f_weight in zip(self.layers, self.style_weights):
            _, C, fh, fw = rendered_feats[layer].shape
            rendered_feats[layer] = rendered_feats[layer].view(B,S, C, fh, fw)
            style_feats[layer] = style_feats[layer].view(B,1, C, fh, fw)
            for i in range(B):
                for j in range(S):
                    style_loss += self.loss_fn(rendered_feats[layer][i,j], style_feats[layer][i,0])*f_weight
        del rendered_feats, style_feats
        return style_loss * self.cfg.weight
    
    def forward(self, prediction, batch, gaussians, depth_dict, global_step):
        if self.cfg.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, prediction, batch, gaussians, depth_dict, global_step
            )
        else:
            return self._forward(prediction, batch, gaussians, depth_dict, global_step)
from dataclasses import dataclass
from pathlib import Path
import gc
import random
from typing import Literal, Optional, Protocol, runtime_checkable, Any

import moviepy.editor as mpy
import torch
import torchvision
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torchvision.transforms import v2 
from loss.loss_lpips import LossLpips
from loss.loss_mse import LossMse
from model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from ..loss.loss_distill import DistillLoss
from src.utils.render import generate_path
from src.utils.point import get_normal_map

from ..loss.loss_huber import HuberLoss, extri_intri_to_pose_encoding

# from model.types import Gaussians

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, abs_relative_difference, delta1_acc
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose, rotation_6d_to_matrix
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
# from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply

from collections import deque

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    cxt_depth_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0
    stylization_start_step: int = 2000
    train_stage: Literal["gs", "style"] = "gs"
    loss_threshold: Optional[float] = None
    clip_loss: bool = False

@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        
        # Set up the model.
        self.encoder_visualizer = None
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses).eval()
        self.stylization_start_step = self.train_cfg.stylization_start_step
        self.train_stage = self.train_cfg.train_stage
        
        if self.model.encoder.pred_pose:
            self.loss_pose = HuberLoss(alpha=self.train_cfg.pose_loss_alpha, delta=self.train_cfg.pose_loss_delta)
        
        if self.model.encoder.distill:
            self.loss_distill = DistillLoss(
                delta=self.train_cfg.pose_loss_delta,
                weight_pose=self.train_cfg.weight_pose,
                weight_depth=self.train_cfg.weight_depth,
                weight_normal=self.train_cfg.weight_normal
            )

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.style_aug_fn = v2.Compose([v2.ColorJitter(brightness=.5, hue=.3)])

        self.loss_window = deque(maxlen=100)  # the most recent 100 losses
        self.dynamic_clip_k = 1.0             # threshold = mean + k*std

    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)
        
    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        # torch.cuda.empty_cache()
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        
        batch: BatchedExample = self.data_shim(batch)
        b, v, c, h, w = batch["context"]["image"].shape

        batch["context"]["image"] = (batch["context"]["image"]+1) / 2
        context_image = batch["context"]["image"].clone()
        batch["context"]["image"] = batch["context"]["image"].view(b*v, c, h, w)
        batch["context"]["image"] = self.style_aug_fn(batch["context"]["image"])
        batch["context"]["image"] = batch["context"]["image"].view(b, v, c, h, w)
        batch["context"]["image"] = (batch["context"]["image"])*2-1

        #stylization_flag = self.global_step > self.stylization_start_step
        stylization_flag = self.train_stage == "style"
        change_style_image =  random.random() > (0.5 + 0.5 * (self.global_step / self.train_cfg.distill_max_steps))
        print(f"Training step {self.global_step:0>6} on Rank {self.global_rank}. "
              f"Stylization flag: {stylization_flag}, Change style image: {change_style_image}, ")
        if not stylization_flag or change_style_image:
            selected_index = random.randint(0, v-1)
            batch["context"]["style_image"] = batch["context"]["image"][:, selected_index, :, :, :].clone()
            batch["context"]["style_image"] = batch["context"]["style_image"].view(b, 1, c, h, w)


        style_image = batch["context"]["style_image"]
        #context_image = (batch["context"]["image"] + 1) / 2
        style_image = (style_image + 1) / 2

        
        # Run the model.
        visualization_dump = None

        encoder_output, output = self.model(context_image, style_image, self.global_step, visualization_dump=visualization_dump)
        gaussians, pred_pose_enc_list, depth_dict = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.depth_dict
        pred_context_pose = encoder_output.pred_context_pose
        infos = encoder_output.infos
        distill_infos = encoder_output.distill_infos
        output.pts_all = encoder_output.pts_all
        output.conf = encoder_output.conf

        num_context_views = pred_context_pose['extrinsic'].shape[1]

        using_index = torch.arange(num_context_views, device=gaussians.means.device)
        batch["using_index"] = using_index
        
        target_gt = (batch["context"]["image"] + 1) / 2
        scene_scale = infos["scene_scale"]
        self.log("train/scene_scale", infos["scene_scale"])
        self.log("train/voxelize_ratio", infos["voxelize_ratio"])

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        if distill_infos is not None:
            consis_absrel = abs_relative_difference(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
            )
            self.log("train/consis_absrel", consis_absrel.mean())

            consis_delta1 = delta1_acc(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
            )
            self.log("train/consis_delta1", consis_delta1.mean())
        else:
            conf_mask = torch.ones_like(output.depth, device=output.depth.device, dtype=torch.bool)
            consis_absrel = abs_relative_difference(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(conf_mask.squeeze(-1), "b v h w -> (b v) h w"),
            )
            self.log("train/consis_absrel", consis_absrel.mean())

            consis_delta1 = delta1_acc(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(conf_mask.squeeze(-1), "b v h w -> (b v) h w"),
            )
            self.log("train/consis_delta1", consis_delta1.mean())            
        
        # Compute and log loss.
        total_loss = 0

        depth_dict['distill_infos'] = distill_infos
        style_loss_names = ["style", "content", "total_variance","adain_style","clip","dinov3content"]
        with torch.amp.autocast('cuda', enabled=False):
            for loss_fn in self.losses:
                if stylization_flag:
                    if not change_style_image and loss_fn.name not in style_loss_names:
                        continue
                else:
                    if loss_fn.name in style_loss_names:
                        continue

                loss = loss_fn.forward(output, batch, gaussians, depth_dict, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                print(f"loss/{loss_fn.name}: {loss.item()}")

                if loss_fn.name == "adain_style":
                    # save adain_style loss
                    adain_loss_val = loss
                else:
                    total_loss = total_loss + loss


            if depth_dict is not None and "depth" in get_cfg()["loss"].keys() and self.train_cfg.cxt_depth_weight > 0:
                depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
                depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
                loss_depth = depth_loss_fn(depth_dict["depth_map"], depth_dict["depth_conf"], batch, cxt_depth_weight=self.train_cfg.cxt_depth_weight)
                self.log("loss/ctx_depth", loss_depth)
                total_loss = total_loss + loss_depth

            if distill_infos is not None:
                # distill ctx pred_pose & depth & normal
                loss_distill_list = self.loss_distill(distill_infos, pred_pose_enc_list, output, batch)
                self.log("loss/distill", loss_distill_list['loss_distill'])
                self.log("loss/distill_pose", loss_distill_list['loss_pose'])
                self.log("loss/distill_depth", loss_distill_list['loss_depth'])
                self.log("loss/distill_normal", loss_distill_list['loss_normal'])
                total_loss = total_loss + loss_distill_list['loss_distill']

        if self.train_stage == "style" and adain_loss_val is not None:
            if torch.isfinite(adain_loss_val):
                total_loss = total_loss + adain_loss_val
                self.log("loss/adain_style_clipped", adain_loss_val)
            else:
                print(f"[Warning] Skip NaN/Inf adain loss at step {self.global_step}")
        self.log("loss/total", total_loss)
        print(f"total_loss: {total_loss}")
        
        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}; "
            )
            
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor
        
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        
        del batch
        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss
    
    def on_after_backward(self):
        total_norm = 0.0
        counter = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                counter += 1
        total_norm = (total_norm / counter) ** 0.5
        print(f"Grad norm: {total_norm:.20f} at step {self.global_step} on Rank {self.global_rank}")
        self.log("loss/grad_norm", total_norm)

        for name, param in self.named_parameters():
            split_name = name.split(".")
            if param.grad is not None and len(split_name) < 3:
                grad_norm = param.grad.detach().data.norm(2)
                self.log(f"grad_norm/{name}", grad_norm.item(), prog_bar=True, logger=True)
        

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):        
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, v, c, h, w = batch["context"]["image"].shape
        batch["context"]["image"] = (batch["context"]["image"]+1) / 2
        context_image = batch["context"]["image"].clone()
        batch["context"]["image"] = batch["context"]["image"].view(b*v, c, h, w)
        batch["context"]["image"] = self.style_aug_fn(batch["context"]["image"])
        batch["context"]["image"] = batch["context"]["image"].view(b, v, c, h, w)
        batch["context"]["image"] = (batch["context"]["image"])*2-1
        assert b == 1
        visualization_dump = {}
        #stylization_flag = self.global_step >= self.stylization_start_step
        stylization_flag = self.train_stage == "style"
        if not stylization_flag:
            selected_index = 0 #random.randint(0, v-1)
            batch["context"]["style_image"] = batch["context"]["image"][:, selected_index, :, :, :].clone()
            batch["context"]["style_image"] = batch["context"]["style_image"].view(b, 1, c, h, w)
        encoder_output, output = self.model(context_image,(batch["context"]["style_image"]+1)/2, 
                                            self.global_step, 
                                            visualization_dump=visualization_dump)
        output.pts_all = encoder_output.pts_all
        output.conf = encoder_output.conf
        gaussians, pred_pose_enc_list, depth_dict = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.depth_dict
        pred_context_pose, distill_infos = encoder_output.pred_context_pose, encoder_output.distill_infos
        infos = encoder_output.infos

        GS_num = infos['voxelize_ratio'] * (h*w*v)
        self.log("val/GS_num", GS_num)
        
        num_context_views = pred_context_pose['extrinsic'].shape[1]
        num_target_views = batch["target"]["extrinsics"].shape[1]
        rgb_pred = output.color[0].float()     
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = (batch["context"]["image"][0].float() + 1) / 2
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # depth metrics
        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_absrel", consis_absrel.mean())
        
        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            valid_mask=rearrange(torch.ones_like(output.depth, device=output.depth.device, dtype=torch.bool), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_delta1", consis_delta1.mean())

        diff_map = torch.abs(output.depth - depth_dict['depth'].squeeze(-1))
        if distill_infos is not None:
            self.log("val/consis_mse", diff_map[distill_infos['conf_mask']].mean())
        else:
            self.log("val/consis_mse", diff_map.mean())
        if (batch_idx) > 5:
            return
        # Construct comparison image.
        context_img = context_image[0] #inverse_normalize(batch["context"]["image"][0])
        context_style_img = inverse_normalize(batch["context"]["style_image"][0])
        # context_img_depth = vis_depth_map(gaussian_means)
        context = []
        context_style = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            if i < context_style_img.shape[0]:
                context_style.append(context_style_img[i])
            else:
                context_style.append(torch.zeros_like(context_style_img[0]))
            # context.append(context_img_depth[i])
        
        colored_diff_map = vis_depth_map(diff_map[0], near=torch.tensor(1e-4, device=diff_map.device), far=torch.tensor(1.0, device=diff_map.device))
        model_depth_pred = depth_dict["depth"].squeeze(-1)[0]
        model_depth_pred = vis_depth_map(model_depth_pred)
        
        # render_normal = (get_normal_map(output.depth.flatten(0, 1), batch["context"]["intrinsics"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.
        # pred_normal = (get_normal_map(depth_dict['depth'].flatten(0, 1).squeeze(-1), batch["context"]["intrinsics"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.
        render_normal = (get_normal_map(output.depth.flatten(0, 1), pred_context_pose["intrinsic"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.
        pred_normal   = (get_normal_map(depth_dict['depth'].flatten(0, 1).squeeze(-1), pred_context_pose["intrinsic"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.


        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            add_label(vcat(*model_depth_pred), "Depth (VGGT Prediction)"),
            add_label(vcat(*pred_normal), "Normal (Prediction)"),
            add_label(vcat(*render_normal), "Normal (VGGT Prediction)"),
            add_label(vcat(*context_style), "Style Context"),
        )
        comparison = comparison
        comparison = torch.nn.functional.interpolate(
            comparison.unsqueeze(0), 
            scale_factor=0.5, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        if stylization_flag:
            _, context_output = self.model(context_image,context_image[:, 0, :, :, :].unsqueeze(1), self.global_step, visualization_dump=visualization_dump)

            context_rgb_pred = context_output.color[0].float()
            sampled_indices = random.sample(range(v), min(4, v))
            rgb_pred_list = [context_style_img[0]]
            context_pred_list = [context_img[0]]
            context_list = [torch.zeros_like(context_style_img[0])]
            for i in sampled_indices:
                rgb_pred_list.append(rgb_pred[i])
                context_list.append(context_img[i])
                context_pred_list.append(context_rgb_pred[i])
                print(f"rgb_pred[{i}] min {rgb_pred[i].min():.4f}, max {rgb_pred[i].max():.4f}, mean {rgb_pred[i].mean():.4f}")
                print(f"context_img[{i}] min {context_img[i].min():.4f}, max {context_img[i].max():.4f}, mean {context_img[i].mean():.4f}")

            results_image = vcat(
                hcat(*rgb_pred_list),
                hcat(*context_pred_list),
                hcat(*context_list),
            )
            results_image = torch.nn.functional.interpolate(
                results_image.unsqueeze(0),
                scale_factor=0.5,
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            self.logger.log_image(
                key="stylization",
                images=[wandb.Image(prep_image(add_border(results_image)), caption=batch["scene"], file_type="jpg")],
                step=self.global_step
            )


        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)
        
        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t, initial_extrinsics, final_extrinsics, initial_intrinsics, final_intrinsics):
            origin_a = initial_extrinsics[:3, 3]
            origin_b = final_extrinsics[:3, 3]
            delta = (origin_a - origin_b).norm()
            extrinsics = generate_wobble(initial_extrinsics, delta * 0.25, t)
            intrinsics = interpolate_intrinsics(initial_intrinsics, final_intrinsics, t)
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)


    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        def trajectory_fn(t, initial_extrinsics, final_extrinsics, initial_intrinsics, final_intrinsics):
            extrinsics = interpolate_extrinsics(initial_extrinsics, final_extrinsics, t)
            intrinsics = interpolate_intrinsics(initial_intrinsics, final_intrinsics, t)
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")


    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t, initial_extrinsics, final_extrinsics, initial_intrinsics, final_intrinsics):
            origin_a = initial_extrinsics[:3, 3]
            origin_b = final_extrinsics[:3, 3]
            delta = (origin_a - origin_b).norm()

            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(initial_extrinsics, final_extrinsics, t * 5 - 2)
            intrinsics = interpolate_intrinsics(initial_intrinsics, final_intrinsics, t * 5 - 2)
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )


    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # 编码，拿到预测相机参数
        encoder_output = self.model.encoder(
            (batch["context"]["image"]+1)/2,
            (batch["context"]["style_image"]+1)/2,
            self.global_step
        )
        gaussians = encoder_output.gaussians
        pred_context_pose = encoder_output.pred_context_pose

        # 用预测的 extrinsics
        initial_extrinsics = pred_context_pose["extrinsic"][0, 0]
        if pred_context_pose["extrinsic"].shape[1] > 1:
            final_extrinsics = pred_context_pose["extrinsic"][0, 1]
        else:
            final_extrinsics = initial_extrinsics

        # 用预测的 intrinsics（如果有）
        if "intrinsic" in pred_context_pose:
            initial_intrinsics = pred_context_pose["intrinsic"][0, 0]
            if pred_context_pose["intrinsic"].shape[1] > 1:
                final_intrinsics = pred_context_pose["intrinsic"][0, 1]
            else:
                final_intrinsics = initial_intrinsics
        else:
            initial_intrinsics = batch["context"]["intrinsics"][0, 0]
            if batch["context"]["intrinsics"].shape[1] > 1:
                final_intrinsics = batch["context"]["intrinsics"][0, 1]
            else:
                final_intrinsics = initial_intrinsics

        # 生成时间序列
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        # 轨迹插值
        extrinsics, intrinsics = trajectory_fn(
            t, initial_extrinsics, final_extrinsics, initial_intrinsics, final_intrinsics
        )

        _, _, _, h, w = batch["context"]["image"].shape
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)

        # 渲染
        output = self.model.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )


    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "interm" in name or "style" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,  # or 1e-4
            end_factor=1.0,
            total_iters=warm_up_steps,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=get_cfg()["trainer"]["max_steps"] - warm_up_steps,
            eta_min=self.optimizer_cfg.lr * 0.001,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_up, cosine_scheduler],
            milestones=[warm_up_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    def on_train_start(self) -> None:
        if self.train_stage == "style":
            # freeze geomtry head
            for name, param in self.named_parameters():
                if "model.encoder.gaussian_param_head" in name:
                    param.requires_grad = False
                    print(f"Freezing parameter: {name}")
        # print out trainable modules
        trainable_modules = set()
        for name, module in self.named_modules():
            split_name = name.split(".")
            if len(split_name) > 3:
                module_name = ".".join(split_name[:3])
            else:
                module_name = name
            if any(param.requires_grad for param in module.parameters()):
                trainable_modules.add(module_name)
        print(f"Trainable modules: {', '.join(trainable_modules)}")
      
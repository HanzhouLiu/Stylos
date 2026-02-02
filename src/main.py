import os
from pathlib import Path
import shutil

import hydra
import torch
import wandb
import random
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DeepSpeedStrategy
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import get_model
from src.misc.weight_modify import checkpoint_filter_fn

import warnings
warnings.filterwarnings("ignore")

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

class HFCheckpointCallback(Callback):
    def __init__(self, output_dir: Path, subdir: str = "hf_checkpoints", keep_last_k: int = 3):
        super().__init__()
        self.hf_dir = Path(output_dir) / subdir
        self.hf_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.saved_paths = []

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        step = trainer.global_step
        save_path = self.hf_dir / f"step_{step:06d}"

        if trainer.global_rank == 0:  # only save on rank 0
            pl_module.model.save_pretrained(save_path)
            print(f"[HFCheckpointCallback] HuggingFace checkpoint saved to {save_path}")

            # Track saved HF checkpoints
            self.saved_paths.append(save_path)

            # If exceeding keep_last_k, remove the oldest
            if len(self.saved_paths) > self.keep_last_k:
                oldest = self.saved_paths.pop(0)
                try:
                    shutil.rmtree(oldest)
                    print(f"[HFCheckpointCallback] Removed old HuggingFace checkpoint {oldest}")
                except Exception as e:
                    print(f"[HFCheckpointCallback] Failed to remove {oldest}: {e}")

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    
    cfg.train.output_path = output_dir
    
    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))
        
        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    
    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Add HuggingFace checkpoint saving
    callbacks.append(HFCheckpointCallback(output_dir))

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    
    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
    
    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        # num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        # strategy="deepspeed_stage_1",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        model,
        get_losses(cfg.loss),
        step_tracker
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    if checkpoint_path is not None:
        print(cyan(f"Loading checkpoint from {checkpoint_path}"))
        model_wrapper.strict_loading= False
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        # remove the state_dict["state_dict"]'s prefix "model."
        model_state_dict = {
            k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
        }
        
        model_wrapper.model.load_state_dict(
            model_state_dict,
            strict=False,
        )

    # if cfg.train.train_stage == "style":
    #     assert checkpoint_path is not None, "Checkpoint path must be provided for style training."
    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module)
        #huggingface_hub_checkpoints_path = output_dir / "hf_checkpoints"
        # create folder
        #huggingface_hub_checkpoints_path.mkdir(parents=True, exist_ok=True)
        #model_name = huggingface_hub_checkpoints_path/"styleanysplat"
        #model_wrapper.model.save_pretrained(model_name)
    elif cfg.mode == "save":
        huggingface_hub_checkpoints_path = output_dir / "hf_checkpoints"
        # create folder
        huggingface_hub_checkpoints_path.mkdir(parents=True, exist_ok=True)
        model_name = huggingface_hub_checkpoints_path/"styleanysplat"
        model_wrapper.model.save_pretrained(model_name)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()

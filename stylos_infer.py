#!/usr/bin/env python3
"""
Stylos inference CLI (images or video)

Examples:

# From image folder (all frames)
python stylos_infer.py \
  --model /path/to/hf_checkpoints/step_015000 \
  --image-folder /path/to/images \
  --style-image examples/unseen_style/test_009_Conroy_Maddox_Surrealism_landscape.jpg \
  --output-dir output/inference

# From video at 1 FPS, first 60 seconds only
python stylos_infer.py \
  --model /path/to/hf_checkpoints/step_015000 \
  --video /path/to/video.mp4 \
  --style-image examples/unseen_style/test_009_Conroy_Maddox_Surrealism_landscape.jpg \
  --output-dir output/inference \
  --extract-fps 1 \
  --max-seconds 60
"""

from __future__ import annotations
import argparse
import os
import sys
import shutil
import math
import tempfile
from pathlib import Path
from typing import List, Optional

import torch

# --- Optional: help Python find your repo modules without installing ---
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_REPO_ROOT) not in sys.path:
    sys.path.append(str(DEFAULT_REPO_ROOT))

# Repo imports (assumed present)
from src.misc.image_io import save_interpolated_video
from src.model.ply_export import export_ply
from src.model.model.stylos import Stylos
from src.utils.image import process_image

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def discover_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not imgs:
        raise FileNotFoundError(f"No images with extensions {IMG_EXTS} found in {folder}")
    return imgs


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(precision: str) -> Optional[torch.dtype]:
    precision = precision.lower()
    if precision == "auto":
        return None  # use autocast when CUDA is available
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def extract_video_frames_1fps(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 1.0,
    max_seconds: Optional[int] = None,
    overwrite: bool = True,
) -> List[Path]:
    """
    Extract frames from `video_path` into `out_dir` at ~target_fps (default 1 FPS).
    Uses OpenCV. Returns sorted list of extracted frame paths.

    Notes:
    - For VFR videos, OpenCV provides best-effort sampling using frame index.
    - If original FPS < target_fps, we sample one frame per original second anyway.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError(
            "OpenCV is required for video input. Please `pip install opencv-python`."
        ) from e

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / orig_fps) if (orig_fps > 0 and total_frames > 0) else None

    if orig_fps <= 0:
        # Fallback: read every frame, drop via time estimation
        print("[WARN] Could not determine video FPS reliably; falling back to frame-count stepping.")

    # Compute frame step
    if orig_fps > 0:
        frame_step = max(1, int(round(orig_fps / max(1e-6, target_fps))))
    else:
        # Unknown FPS; just take every 30th frame as ~1 FPS guess if target_fps=1
        frame_step = max(1, int(round(30.0 / max(1e-6, target_fps))))

    # If limiting by seconds, cap the number of frames we intend to read
    max_frames = None
    if max_seconds is not None and orig_fps > 0:
        max_frames = int(max_seconds * orig_fps)

    frame_idx = 0
    saved = 0
    out_paths: List[Path] = []
    while True:
        ret = cap.grab()
        if not ret:
            break
        # Save every `frame_step`-th frame
        if frame_idx % frame_step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break
            # Convert BGR->RGB if your process_image expects RGB; but since we save to disk,
            # we'll keep OpenCV's BGR and let process_image handle reading & conversion.
            out_path = out_dir / f"frame_{frame_idx:09d}.png"
            cv2.imwrite(str(out_path), frame)
            out_paths.append(out_path)
            saved += 1
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    if not out_paths:
        raise RuntimeError("No frames were extracted from the video (check codec/FPS).")
    return sorted(out_paths)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stylos inference CLI (images or video)")

    io = p.add_argument_group("Inputs")
    src = io.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-folder", type=Path, help="Folder with input RGB images")
    src.add_argument("--video", type=Path, help="Path to an input video")
    io.add_argument("--style-image", required=True, type=Path, help="Path to the style image")

    video_grp = p.add_argument_group("Video extraction (if --video is used)")
    video_grp.add_argument("--extract-fps", type=float, default=1.0, help="Target extraction FPS (default: 1.0)")
    video_grp.add_argument("--max-seconds", type=int, default=None, help="Optional upper bound on duration to use")
    video_grp.add_argument("--keep-frames", action="store_true", help="Do not delete temp frames after running")

    out = p.add_argument_group("Outputs")
    out.add_argument("--output-dir", type=Path, default=Path("output/inference"), help="Output base directory")

    perf = p.add_argument_group("Performance")
    perf.add_argument("--num-views", type=int, default=-1, help="Max number of frames/images to use (-1=all)")
    perf.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device selection")
    perf.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                      help="Computation precision (auto uses autocast on CUDA)")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    misc.add_argument("--repo-root", type=Path, default=None, help="Optional repo root to append to sys.path")
    return p


def main(args) -> None:
    # Optional: add a custom repo root to sys.path if provided
    if args.repo_root is not None and str(args.repo_root) not in sys.path:
        sys.path.append(str(args.repo_root))

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    target_dtype = pick_dtype(args.precision)

    # Load model
    model = Stylos.from_pretrained(str(args.model)) if hasattr(args, "model") else None
    # Keep original interface: require --model
    if model is None:
        raise ValueError("--model is required")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Acquire frames: from images or from video
    temp_dir = None
    try:
        if args.video:
            if not args.video.exists():
                raise FileNotFoundError(f"Video not found: {args.video}")
            temp_dir = Path(tempfile.mkdtemp(prefix="stylos_frames_"))
            print(f"[INFO] Extracting frames at {args.extract_fps} FPS from {args.video} -> {temp_dir}")
            extracted = extract_video_frames_1fps(
                args.video, temp_dir, target_fps=args.extract_fps, max_seconds=args.max_seconds
            )
            all_imgs = extracted
        else:
            all_imgs = [str(p) for p in discover_images(args.image_folder)]

        # Trim to num-views
        if args.num_views is None or args.num_views < 0 or args.num_views > len(all_imgs):
            imgs_pick = all_imgs
            print(f"[INFO] Using all {len(imgs_pick)} frames")
        else:
            imgs_pick = all_imgs[: args.num_views]
            if len(imgs_pick) < args.num_views :
                print(f"[WARN] Only found {len(imgs_pick)} images, less than requested {args.num_views }.")

        # Load tensors (assumes process_image -> [-1, 1])
        images = [process_image(str(p)) for p in imgs_pick]
        images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, H, W]
        b, v, _, h, w = images.shape

        # Style image
        if not args.style_image.exists():
            raise FileNotFoundError(f"Style image not found: {args.style_image}")
        style_tensor = process_image(str(args.style_image))
        style_image = torch.stack([style_tensor], dim=0).unsqueeze(0).to(device)  # [1,1,3,H,W]
        # Output path
        style_name =  args.style_image.stem
        if args.video:
            save_folder_name = f"{style_name}__{args.video.stem}"
        if args.image_folder:
            save_folder_name = f"{style_name}__{args.image_folder.name}"
        save_path = args.output_dir / save_folder_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Inference (convert [-1,1] -> [0,1])
        inputs_01 = (images + 1) * 0.5
        style_01 = (style_image + 1) * 0.5

        use_autocast = (args.precision == "auto" and device.type == "cuda") or \
                       (target_dtype in (torch.float16, torch.bfloat16))
        autocast_dtype = None
        if target_dtype == torch.float16:
            autocast_dtype = torch.float16
        elif target_dtype == torch.bfloat16:
            autocast_dtype = torch.bfloat16

        with torch.no_grad():
            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    gaussians, pred_context_pose = model.inference(inputs_01, style_image=style_01)
            else:
                if target_dtype == torch.float32:
                    inputs_01 = inputs_01.float()
                    style_01 = style_01.float()
                gaussians, pred_context_pose = model.inference(inputs_01, style_image=style_01)

        # Save outputs
        pred_all_extrinsic = pred_context_pose["extrinsic"]
        pred_all_intrinsic = pred_context_pose["intrinsic"]

        save_interpolated_video(
            pred_all_extrinsic,
            pred_all_intrinsic,
            b, h, w,
            gaussians,
            str(save_path),
            model.decoder
        )

        export_ply(
            gaussians.means[0],
            gaussians.scales[0],
            gaussians.rotations[0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            save_path / "gaussians.ply"
        )

        print(f"[OK] Saved video & PLY to: {save_path}")
        print(f" - Views used: {len(imgs_pick)} (from {'video' if args.video else args.image_folder})")
        print(f" - Style: {args.style_image} (style-name='{style_name}')")
        print(f" - Device: {device}, Precision: {args.precision}")

    finally:
        # Clean temp frames unless asked to keep
        if temp_dir and (not args.keep_frames):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    parser = build_argparser()
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to HF checkpoint directory for Stylos.from_pretrained"
    )
    args = parser.parse_args()
    main(args)

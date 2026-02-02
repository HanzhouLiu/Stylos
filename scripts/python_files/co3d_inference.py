from pathlib import Path
import torch
import os
import argparse
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import time

from src.model.model.anysplat import AnySplat


# -------------------------
# Image preprocessing
# -------------------------
def process_image(img_path, target_size=448):
    """
    Load an image, convert to RGB if needed, resize while preserving aspect ratio,
    then center-crop to the target size (default: 448x448).
    Normalize to [-1, 1] and return a tensor.
    """
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    width, height = img.size
    if width > height:
        new_height = target_size
        new_width = int(width * (new_height / height))
    else:
        new_width = target_size
        new_height = int(height * (new_width / width))
    img = img.resize((new_width, new_height), Image.BICUBIC)

    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    img = img.crop((left, top, right, bottom))

    img_tensor = T.ToTensor()(img) * 2.0 - 1.0
    return img_tensor


# -------------------------
# Save output frames
# -------------------------
def save_output_frames(gaussians, extrinsics, intrinsics, decoder,
                       save_dir, near, far, h, w, prefix="output_",
                       render_batch=0, start_index=0):
    """
    Render frames from the predicted Gaussian scene representation
    and save them as PNG images in save_dir.
    start_index ensures global indexing across batches.
    """
    os.makedirs(save_dir, exist_ok=True)
    b, v, _, _ = extrinsics.shape
    indices = list(range(v))

    if render_batch <= 0:
        # Render all frames at once
        output = decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        batches = [(indices, output)]
    else:
        # Split into smaller batches
        batches = []
        for i in range(0, v, render_batch):
            idx = indices[i:i + render_batch]
            extr = extrinsics[:, idx]
            intr = intrinsics[:, idx]
            nr = near[:, idx]
            fr = far[:, idx]
            out = decoder.forward(
                gaussians, extr, intr, nr, fr, (h, w), "depth"
            )
            batches.append((idx, out))

    # Save images batch by batch
    for idxs, out in batches:
        for j, frame_idx in enumerate(idxs):
            img = out.color[0, j].detach().cpu()
            if img.min() < 0:
                img = (img + 1) * 0.5
            img = (img * 255).clamp(0, 255).byte()
            img = img.permute(1, 2, 0).numpy()
            global_idx = start_index + frame_idx
            Image.fromarray(img).save(
                os.path.join(save_dir, f"{prefix}{global_idx:03d}.png")
            )


def save_gt_frames(content_tensors, gt_dir, scene_name, prefix="output_"):
    """
    Save the processed content images (GT) into gt_dir/{scene_name}/images/.
    File names follow the same pattern as rendered results (output_000.png).
    If the directory already exists, skip saving.
    """
    save_dir = Path(gt_dir) / scene_name / "images"
    if save_dir.exists():
        print(f"[INFO] GT frames already exist at {save_dir}, skipping save.")
        return

    os.makedirs(save_dir, exist_ok=True)

    for idx, img in enumerate(content_tensors):
        img = (img + 1) * 0.5  # back to [0,1]
        img = (img * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(save_dir / f"{prefix}{idx:03d}.png")

    print(f"[INFO] Saved GT frames to {save_dir}")


# -------------------------
# Main inference
# -------------------------
def run_inference(checkpoint, content_dir, style_dir, exp_name,
                  max_frames=None, frame_stride=1, render_batch=0,
                  batch_size=18,
                  gt_dir=None, out_dir="output/inference"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading checkpoint from {checkpoint}")
    model = AnySplat.from_pretrained(str(checkpoint), local_files_only=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load content images
    image_files = sorted([
        os.path.join(content_dir, f)
        for f in os.listdir(content_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    assert len(image_files) > 0, f"No images found in {content_dir}"

    # Apply stride and frame limit
    image_files = image_files[::frame_stride]
    if max_frames is not None:
        image_files = image_files[:max_frames]

    print(f"[INFO] Total {len(image_files)} content frames (stride={frame_stride}, max={max_frames})")

    # scene_name
    scene_parent = Path(content_dir).parent
    category = scene_parent.parent.name
    scene_name = f"{category}_{scene_parent.name}"

    # save GT frames if needed
    if gt_dir is not None:
        content_tensors = [process_image(p) for p in image_files]
        save_gt_frames(content_tensors, gt_dir, scene_name)

    # load styles
    style_files = sorted([
        os.path.join(style_dir, f)
        for f in os.listdir(style_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    assert len(style_files) > 0, f"No style images found in {style_dir}"

    base_dir = os.path.join(out_dir, exp_name, scene_name, "styles")

    print(f"[INFO] Running per-style inference ({len(style_files)} styles, batch_size={batch_size})")

    # split into batches
    for start in range(0, len(image_files), batch_size):
        batch_files = image_files[start:start+batch_size]
        content_tensors = [process_image(p) for p in batch_files]
        images_tensor = torch.stack(content_tensors, dim=0).unsqueeze(0).to(device)  # [1, v, 3, H, W]
        b, v, _, h, w = images_tensor.shape

        for idx, sfile in enumerate(style_files):
            sname = Path(sfile).stem
            out_dir_style = os.path.join(base_dir, sname, "images")
            os.makedirs(out_dir_style, exist_ok=True)

            style_tensor = process_image(sfile).unsqueeze(0).unsqueeze(0).to(device)

            print(f"[STYLE {idx+1}/{len(style_files)}] {sname}, batch {start}-{start+v-1}")
            start_time = time.time()
            gaussians, pred_context_pose = model.inference(
                (images_tensor + 1) * 0.5, style_image=(style_tensor + 1) * 0.5
            )
            elapsed = time.time() - start_time
            print(f"  -> Inference time: {elapsed:.2f} s")

            save_output_frames(
                gaussians,
                pred_context_pose["extrinsic"],
                pred_context_pose["intrinsic"],
                model.decoder,
                out_dir_style,
                torch.full((1, v), 0.1, device=device),
                torch.full((1, v), 100.0, device=device),
                h,
                w,
                prefix="output_",
                render_batch=render_batch,
                start_index=start
            )


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run StyleAnySplat inference per style image (saving frames + optional GT content)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--content_dir", type=str, required=True,
                        help="Directory containing content images")
    parser.add_argument("--style_dir", type=str, required=True,
                        help="Directory containing style images")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name for saving results")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Use only the first N frames (default: all)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Sample every N-th frame (default: 1)")
    parser.add_argument("--render_batch", type=int, default=0,
                        help="Number of frames to render per batch (default: 0 = all at once)")
    parser.add_argument("--batch_size", type=int, default=18,
                        help="Number of content frames per inference batch (default: 16)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Optional directory to save processed GT content frames")
    parser.add_argument("--out_dir", type=str, default="output/inference",
                        help="Base directory for saving inference results")

    args = parser.parse_args()

    run_inference(
        checkpoint=args.checkpoint,
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        exp_name=args.exp_name,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        render_batch=args.render_batch,
        batch_size=args.batch_size,
        gt_dir=args.gt_dir,
        out_dir=args.out_dir,
    )
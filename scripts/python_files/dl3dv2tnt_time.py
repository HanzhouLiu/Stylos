from pathlib import Path
import torch
import os
import argparse
from PIL import Image
import torchvision.transforms as T

from src.model.model.anysplat import AnySplat
from src.misc.image_io import save_interpolated_video
import time


# -------------------------
# Image preprocessing
# -------------------------
def process_image(img_path, target_size=448):
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



def save_output_frames(gaussians, extrinsics, intrinsics, decoder,
                       save_dir, near, far, h, w, prefix="output_"):
    os.makedirs(save_dir, exist_ok=True)
    output = decoder.forward(
        gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
    )
    b, v, _, _ = extrinsics.shape
    for j in range(v):
        img = output.color[0, j].detach().cpu()
        if img.min() < 0:
            img = (img + 1) * 0.5
        img = (img * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        Image.fromarray(img).save(os.path.join(save_dir, f"{prefix}{j:03d}.png"))


# -------------------------
# Rendering with batching
# -------------------------
def render_images(checkpoint, content_dir, style_dir, save_dir="render_outputs",
                  max_content_frames=32, num_styles=10, batch_size=16):

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
    image_files = image_files[:max_content_frames]
    print(f"[INFO] Using {len(image_files)} content frames")

    # Load style images
    style_files = sorted([
        os.path.join(style_dir, f)
        for f in os.listdir(style_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    style_files = style_files[:num_styles]
    print(f"[INFO] Using {len(style_files)} style images")

    # -------------------------
    # Timing setup
    # -------------------------
    warmup_frames = 48  
    timing_frames = 112  
    infer_times = []
    processed_frames = 0 

    global_index = 0 

    # -------------------------
    # Main loop
    # -------------------------
    for start in range(0, len(image_files), batch_size):
        batch_files = image_files[start:start + batch_size]
        content_tensors = [process_image(p) for p in batch_files]
        images_tensor = torch.stack(content_tensors, dim=0).unsqueeze(0).to(device)
        b, v, _, h, w = images_tensor.shape

        print(f"\n[INFO] Processing content frames {start}–{start+v-1}")

        for sfile in style_files:
            style_name = Path(sfile).stem
            style_tensor = process_image(sfile).unsqueeze(0).unsqueeze(0).to(device)

            print(f"[STYLE] Rendering {style_name} for frames {start}-{start+v-1}")

            torch.cuda.synchronize()
            t0 = time.time()

            gaussians, pred_context_pose = model.inference(
                (images_tensor + 1) * 0.5,
                style_image=(style_tensor + 1) * 0.5
            )

            output = model.decoder.forward(
                gaussians,
                pred_context_pose["extrinsic"],
                pred_context_pose["intrinsic"],
                torch.full((1, v), 0.1, device=device),
                torch.full((1, v), 100.0, device=device),
                (h, w),
                "depth"
            )

            torch.cuda.synchronize()
            total_time = time.time() - t0 

            per_frame_time = total_time / v

            for _ in range(v):
                if processed_frames >= warmup_frames:
                    infer_times.append(per_frame_time)
                processed_frames += 1

            save_path = os.path.join(save_dir, style_name)
            os.makedirs(save_path, exist_ok=True)

            for j in range(v):
                img = output.color[0, j].detach().cpu()
                if img.min() < 0:
                    img = (img + 1) * 0.5
                img = (img * 255).clamp(0, 255).byte()
                img = img.permute(1, 2, 0).numpy()

                fname = f"output_{global_index:03d}.png"
                Image.fromarray(img).save(os.path.join(save_path, fname))

                global_index += 1

    if len(infer_times) > 0:
        avg_time = sum(infer_times) / len(infer_times)
        print("\n==============================")
        print(f" Warmup frames      : {warmup_frames}")
        print(f" Timed frames       : {len(infer_times)} (expected {timing_frames})")
        print(f" Avg time per frame : {avg_time:.6f} sec")
        print(f" FPS                : {1.0 / avg_time:.2f}")
        print("==============================\n")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render StyleAnySplat outputs (PNG + MP4) with batching"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--content_dir", type=str, required=True,
                        help="Directory containing content images")
    parser.add_argument("--style_dir", type=str, required=True,
                        help="Directory containing style images")
    parser.add_argument("--save_dir", type=str, default="render_outputs",
                        help="Directory to save rendered output images")
    parser.add_argument("--max_content_frames", type=int, default=160,
                        help="Max number of content frames")
    parser.add_argument("--num_styles", type=int, default=1,
                        help="Number of styles to render")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of content frames per batch")

    args = parser.parse_args()

    render_images(
        checkpoint=args.checkpoint,
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        save_dir=args.save_dir,
        max_content_frames=args.max_content_frames,
        num_styles=args.num_styles,
        batch_size=args.batch_size,
    )

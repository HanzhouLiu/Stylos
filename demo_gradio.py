#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import torch
from PIL import Image

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # adjust if needed
sys.path.append(str(PROJECT_ROOT))

from src.misc.image_io import save_interpolated_video
from src.model.model.stylos import Stylos
from src.model.ply_export import export_ply
from src.utils.image import process_image

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
TMP_ROOT = Path("demo_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    [None, "examples/video/re10k_1eca36ec55b88fe4.mp4", "examples/demo_styles/23.jpeg", "re10k", "1eca36ec55b88fe4", "2", "Real", "True"],
    [None, "examples/video/bungeenerf_colosseum.mp4", "examples/demo_styles/24.jpeg", "bungeenerf", "colosseum", "8", "Synthetic", "True"],
    [None, "examples/video/fox.mp4", "examples/demo_styles/201.png", "InstantNGP", "fox", "14", "Real", "True"],
    [None, "examples/video/matrixcity_street.mp4", "examples/demo_styles/201.png", "matrixcity", "street", "32", "Synthetic", "True"],
    [None, "examples/video/vrnerf_apartment.mp4", "examples/demo_styles/977.png", "vrnerf", "apartment", "32", "Real", "True"],
    [None, "examples/video/vrnerf_kitchen.mp4", "examples/demo_styles/1098.png", "vrnerf", "kitchen", "17", "Real", "True"],
    [None, "examples/video/vrnerf_riverview.mp4", "examples/demo_styles/00054987.png", "vrnerf", "riverview", "12", "Real", "True"],
    [None, "examples/video/vrnerf_workshop.mp4", "examples/demo_styles/1842.png", "vrnerf", "workshop", "32", "Real", "True"],
    [None, "examples/video/fillerbuster_ramen.mp4", "examples/demo_styles/2190.png", "fillerbuster", "ramen", "32", "Real", "True"],
    [None, "examples/video/meganerf_rubble.mp4", "examples/demo_styles/00011395.png", "meganerf", "rubble", "10", "Real", "True"],
    [None, "examples/video/llff_horns.mp4", "examples/demo_styles/00018289.png", "llff", "horns", "12", "Real", "True"],
    [None, "examples/video/llff_fortress.mp4", "examples/demo_styles/00047052.png", "llff", "fortress", "7", "Real", "True"],
    [None, "examples/video/dtu_scan_106.mp4", "examples/demo_styles/1414.png", "dtu", "scan_106", "20", "Real", "True"],
    [None, "examples/video/horizongs_hillside_summer.mp4", "examples/demo_styles/00091988.png", "horizongs", "hillside_summer", "55", "Synthetic", "True"],
    [None, "examples/video/kitti360.mp4", "examples/demo_styles/00069352.png", "kitti360", "kitti360", "64", "Real", "True"],
]

# ---------------------------------------------------------------------
# Types / Model holder
# ---------------------------------------------------------------------
@dataclass
class ModelBundle:
    stylos_model: Stylos
    torch_device: torch.device


# ---------------------------------------------------------------------
# Utility: directory & IO helpers
# ---------------------------------------------------------------------
def create_run_dir(base_dir: Path = TMP_ROOT) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_path = base_dir / f"run_{ts}"
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def ensure_subdir(parent_dir: Path, name: str, clear_existing: bool = False) -> Path:
    target = parent_dir / name
    if clear_existing and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Upload handlers
# ---------------------------------------------------------------------
def ingest_images_or_video(
    video_input: Optional[str],
    image_inputs: Optional[List[str]],
    reuse_dir: Optional[Path] = None,
) -> Tuple[Path, List[Path]]:
    """
    Create/prepare target_dir/images and store inputs. Returns (target_dir, images).
    """
    empty_cuda_cache()
    target_dir = reuse_dir if (reuse_dir and reuse_dir.exists()) else create_run_dir()
    images_dir = ensure_subdir(target_dir, "images", clear_existing=True)

    saved_paths: List[Path] = []

    # images
    if image_inputs:
        for item in image_inputs:
            src = Path(item if isinstance(item, str) else item["name"])
            dst = images_dir / src.name
            shutil.copy(str(src), str(dst))
            saved_paths.append(dst)

    # video
    if video_input:
        video_path = Path(video_input if isinstance(video_input, str) else video_input["name"])
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps * 1))  # 1 frame/sec
        frame_idx = 0
        saved_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % interval == 0:
                out_path = images_dir / f"{saved_idx:06}.png"
                cv2.imwrite(str(out_path), frame)
                saved_paths.append(out_path)
                saved_idx += 1
        cap.release()

    saved_paths.sort()
    return target_dir, saved_paths


def ingest_style_image(style_img: Optional[str], reuse_dir: Optional[Path]) -> Tuple[Path, List[Path]]:
    """
    Store a single style image at target_dir/styles/style.jpg.
    """
    target_dir = reuse_dir if (reuse_dir and reuse_dir.exists()) else create_run_dir()
    styles_dir = ensure_subdir(target_dir, "styles", clear_existing=True)

    paths: List[Path] = []
    if style_img:
        dst = styles_dir / "style.jpg"
        Image.open(style_img).convert("RGB").save(str(dst))
        paths.append(dst)

    return target_dir, paths


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
def run_reconstruction(
    target_dir: Path,
    bundle: ModelBundle,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (ply_path, rgb_video_path, depth_video_path)
    """
    if not target_dir.exists():
        return None, None, None

    empty_cuda_cache()

    images_dir = target_dir / "images"
    style_path = target_dir / "styles" / "style.jpg"
    if not images_dir.exists() or not style_path.exists():
        return None, None, None

    image_files = sorted([str(images_dir / f) for f in os.listdir(images_dir)])
    image_tensors = [process_image(p) for p in image_files]
    input_tensor = torch.stack(image_tensors, dim=0).unsqueeze(0).to(bundle.torch_device)  # [1,K,3,448,448]
    batch, views, channels, height, width = input_tensor.shape
    assert channels == 3, "Images must have 3 channels."

    style_tensor = process_image(str(style_path)).unsqueeze(0).unsqueeze(0).to(bundle.torch_device)  # [1,1,3,448,448]

    with torch.no_grad():
        gauss, pose_dict = bundle.stylos_model.inference(
            (input_tensor + 1) * 0.5, style_image=(style_tensor + 1) * 0.5
        )

    pred_extr = pose_dict["extrinsic"]
    pred_intr = pose_dict["intrinsic"]

    rgb_path, depth_path = save_interpolated_video(
        pred_extr,
        pred_intr,
        batch,
        height,
        width,
        gauss,
        str(target_dir),
        bundle.stylos_model.decoder,
    )

    ply_path = target_dir / "gaussians.ply"
    export_ply(
        gauss.means[0],
        gauss.scales[0],
        gauss.rotations[0],
        gauss.harmonics[0],
        gauss.opacities[0],
        ply_path,
        save_sh_dc_only=True,
    )

    empty_cuda_cache()
    return str(ply_path), rgb_path, depth_path


# ---------------------------------------------------------------------
# UI callbacks (pure functions; no globals)
# ---------------------------------------------------------------------
def cb_update_content_on_video(
    video_input,
    current_dir_str: str,
    style_ready_flag: bool,
):
    """
    Authoritative video handler:
    - Clears/overwrites images dir
    - Ignores any existing images input
    - Clears the images file component on the UI
    """
    current_dir = Path(current_dir_str) if current_dir_str and current_dir_str != "None" else None
    reuse = current_dir if (style_ready_flag and current_dir and current_dir.exists()) else None

    # No new video? Clear gallery & disable submit.
    if not video_input:
        if current_dir and (current_dir / "images").exists():
            shutil.rmtree(current_dir / "images")
        return (
            None,                              # 3D viewer
            current_dir_str or "None",         # target dir text
            [],                                # gallery
            False,                             # content_ready_state
            gr.update(interactive=False),      # reconstruct_btn
            gr.update(value=None),             # images_input_comp <- clear
        )

    # Ingest ONLY video frames (authoritative)
    run_dir, saved = ingest_images_or_video(video_input, None, reuse_dir=reuse)
    can_submit = bool(saved) and style_ready_flag

    return (
        None,
        str(run_dir),
        [str(p) for p in saved],
        bool(saved),
        gr.update(interactive=can_submit),
        gr.update(value=None),  # clear images input so it can't “stick”
    )


def cb_update_content_on_images(
    images_input,
    current_dir_str: str,
    style_ready_flag: bool,
):
    """
    Authoritative images handler:
    - Clears/overwrites images dir
    - Ignores any existing video input
    - Clears the video component on the UI
    """
    current_dir = Path(current_dir_str) if current_dir_str and current_dir_str != "None" else None
    reuse = current_dir if (style_ready_flag and current_dir and current_dir.exists()) else None

    # No new images? Clear gallery & disable submit.
    if not images_input:
        if current_dir and (current_dir / "images").exists():
            shutil.rmtree(current_dir / "images")
        return (
            None,                              # 3D viewer
            current_dir_str or "None",         # target dir text
            [],                                # gallery
            False,                             # content_ready_state
            gr.update(interactive=False),      # reconstruct_btn
            gr.update(value=None),             # video_input_comp <- clear
        )

    # Ingest ONLY images (authoritative)
    run_dir, saved = ingest_images_or_video(None, images_input, reuse_dir=reuse)
    can_submit = bool(saved) and style_ready_flag

    return (
        None,
        str(run_dir),
        [str(p) for p in saved],
        bool(saved),
        gr.update(interactive=can_submit),
        gr.update(value=None),  # clear video input so it can't “stick”
    )

def cb_update_style(
    style_path,
    current_dir_str: str,
    content_ready_flag: bool,
) -> Tuple[str, bool, gr.update]:
    current_dir = Path(current_dir_str) if current_dir_str and current_dir_str != "None" else None
    run_dir, style_paths = ingest_style_image(style_path, reuse_dir=current_dir)
    is_ready = len(style_paths) > 0
    return str(run_dir), is_ready, gr.update(interactive=(is_ready and content_ready_flag))


def cb_example_loader(
    images_input,
    video_input,
    style_input,
    dataset_name,
    scene_name,
    num_images_str,
    image_type,
    is_example_flag,
) -> Tuple[None, None, None, str, List[str], str, bool, bool, gr.update]:
    run_dir, saved_imgs = ingest_images_or_video(video_input, None, reuse_dir=None)
    run_dir, style_imgs = ingest_style_image(style_input, reuse_dir=run_dir)
    content_ok = len(saved_imgs) > 0
    style_ok = len(style_imgs) > 0
    submit_ok = gr.update(interactive=(content_ok and style_ok))
    style_one = str(style_imgs[0]) if style_imgs else None
    return None, None, None, str(run_dir), [str(p) for p in saved_imgs], style_one, content_ok, style_ok, submit_ok


def cb_clear_states() -> Tuple[bool, bool, gr.update]:
    return False, False, gr.update(interactive=False)


def cb_clear_outputs():
    return None, None, None


def cb_reconstruct(
    target_dir_str: str,
    model_state: ModelBundle,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not target_dir_str or target_dir_str == "None":
        return None, None, None
    tdir = Path(target_dir_str)
    start = time.time()
    ply, rgb, depth = run_reconstruction(tdir, model_state)
    print(f"Reconstruction total time: {time.time() - start:.2f}s")
    return ply, rgb, depth


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------
def build_theme_and_css():
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )
    css = """
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }
        .example-log * {
            font-style: italic;
            font-size: 16px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }
        #my_radio .wrap { display: flex; flex-wrap: nowrap; justify-content: center; align-items: center; }
        #my_radio .wrap label {
            display: flex; width: 50%; justify-content: center; align-items: center;
            margin: 0; padding: 10px 0; box-sizing: border-box;
        }
    """
    return theme, css


def load_model_bundle(checkpoint_dir: str) -> ModelBundle:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = Stylos.from_pretrained(checkpoint_dir)
    model_instance = model_instance.to(dev)
    model_instance.eval()
    for p in model_instance.parameters():
        p.requires_grad = False
    return ModelBundle(stylos_model=model_instance, torch_device=dev)


def create_interface(model_state: ModelBundle) -> gr.Blocks:
    theme, css = build_theme_and_css()

    with gr.Blocks(css=css, title="Stylos Demo", theme=theme) as demo_app:
        gr.Markdown("<h1 style='text-align: center;'>Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting</h1>")

        with gr.Row():
            gr.Markdown(
                """
                <p align="center">
                    <a title="Website" href="https://github.com/HanzhouLiu/StylOS" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                    </a>
                    <a title="arXiv" href="https://arxiv.org/abs/2509.26455" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                    </a>
                    <a title="Github" href="https://github.com/HanzhouLiu/StylOS" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/Github-Page-black" alt="badge-github-stars">
                    </a>
                </p>
                """
            )

        with gr.Row():
            gr.Markdown(
                """
                ### Getting Started:
                1. **Upload Content Data:** Use "Upload Video" or "Upload Images". Videos are split at 1 FPS.
                2. **Preview:** Uploaded images appear in the gallery.
                3. **Upload Style Image:** Upload a single style image.
                4. **Stylize & Reconstruct:** Click **Reconstruct** to run 3D reconstruction.
                5. **Visualize:** View 3D Gaussian splats, along with rendered RGB and depth videos.
                <strong style="color: #0ea5e9;">Note:</strong>
                <span style="color: #0ea5e9; font-weight: bold;">
                Generated splats can be large; if the viewer fails on HF, download the .ply and use SuperSplat or other viewers.
                </span>
                """
            )

        # Hidden/state components
        run_dir_text = gr.Textbox(label="Target Dir", visible=False, value="None")
        is_example_text = gr.Textbox(label="is_example", visible=False, value="None")
        num_images_text = gr.Textbox(label="num_images", visible=False, value="None")
        dataset_text = gr.Textbox(label="dataset_name", visible=False, value="None")
        scene_text = gr.Textbox(label="scene_name", visible=False, value="None")
        image_type_text = gr.Textbox(label="image_type", visible=False, value="None")

        content_ready_state = gr.State(False)
        style_ready_state = gr.State(False)
        model_state_holder = gr.State(model_state)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Video Data"):
                        video_input_comp = gr.Video(label="Upload Video", interactive=True)
                    with gr.Tab("Images Data"):
                        images_input_comp = gr.File(file_count="multiple", label="Upload Images", interactive=True)

                    gallery_comp = gr.Gallery(
                        label="Content Preview",
                        columns=4,
                        height="300px",
                        show_download_button=True,
                        object_fit="contain",
                        preview=True,
                    )

                    style_input_comp = gr.Image(type="filepath", label="Upload Style Image", interactive=True)

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Stylos Output"):
                        with gr.Column():
                            model3d_comp = gr.Model3D(
                                label="3D Reconstructed Gaussian Splat",
                                height=540,
                                zoom_speed=0.5,
                                pan_speed=0.5,
                                camera_position=[20, 20, 20],
                            )
                        with gr.Row():
                            rgb_video_comp = gr.Video(label="RGB Video", interactive=False, autoplay=True)
                            depth_video_comp = gr.Video(label="Depth Video", interactive=False, autoplay=True)

                        with gr.Row():
                            reconstruct_btn = gr.Button("Reconstruct", scale=1, variant="primary", interactive=False)
                            clear_btn = gr.ClearButton(
                                [
                                    video_input_comp,
                                    images_input_comp,
                                    style_input_comp,
                                    model3d_comp,
                                    run_dir_text,
                                    gallery_comp,
                                    rgb_video_comp,
                                    depth_video_comp,
                                ],
                                scale=1,
                            )

        # Examples
        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

        gr.Examples(
            examples=EXAMPLES,
            inputs=[
                images_input_comp,
                video_input_comp,
                style_input_comp,
                dataset_text,
                scene_text,
                num_images_text,
                image_type_text,
                is_example_text,
            ],
            outputs=[
                model3d_comp,
                rgb_video_comp,
                depth_video_comp,
                run_dir_text,
                gallery_comp,
                style_input_comp,
                content_ready_state,
                style_ready_state,
                reconstruct_btn,
            ],
            fn=cb_example_loader,
            cache_examples=False,
            examples_per_page=50,
            run_on_click=True,
        )

        gr.Markdown("<p style='text-align: center; font-style: italic; color: #666;'>We thank VGGT and AnySplat for their excellent gradio implementation!</p>")

        # Wiring
        reconstruct_btn.click(
            fn=cb_clear_outputs,
            inputs=[],
            outputs=[model3d_comp, rgb_video_comp, depth_video_comp],
        ).then(
            fn=cb_reconstruct,
            inputs=[run_dir_text, model_state_holder],
            outputs=[model3d_comp, rgb_video_comp, depth_video_comp],
        ).then(
            fn=lambda: "False",
            inputs=[],
            outputs=[is_example_text],
        )

        video_input_comp.change(
            fn=cb_update_content_on_video,
            inputs=[video_input_comp, run_dir_text, style_ready_state],
            outputs=[
                model3d_comp,      # cleared
                run_dir_text,      # set
                gallery_comp,      # set
                content_ready_state,
                reconstruct_btn,   # enabled/disabled
                images_input_comp, # <-- clear the images file input
            ],
        )

        images_input_comp.change(
            fn=cb_update_content_on_images,
            inputs=[images_input_comp, run_dir_text, style_ready_state],
            outputs=[
                model3d_comp,     # cleared
                run_dir_text,     # set
                gallery_comp,     # set
                content_ready_state,
                reconstruct_btn,  # enabled/disabled
                video_input_comp, # <-- clear the video input
            ],
        )


        style_input_comp.change(
            fn=cb_update_style,
            inputs=[style_input_comp, run_dir_text, content_ready_state],
            outputs=[run_dir_text, style_ready_state, reconstruct_btn],
        )

        clear_btn.click(
            fn=cb_clear_states,
            inputs=[],
            outputs=[content_ready_state, style_ready_state, reconstruct_btn],
        )

    return demo_app


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    # Server parameters
    launch_share = True
    server_host = "127.0.0.1"
    server_port = None  # or int

    # Load model from your checkpoint
    checkpoint_path = "stylos_hf_chk"
    model_bundle = load_model_bundle(checkpoint_path)

    interface = create_interface(model_bundle)
    interface.queue(max_size=20).launch(
        show_error=True,
        share=launch_share,
        server_name=server_host,
        server_port=server_port,
    )


if __name__ == "__main__":
    main()

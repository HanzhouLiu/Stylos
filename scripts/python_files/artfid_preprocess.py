# Python snippets for copying image files and forming folders for calculating ArtFID (StyleID)
# Then, refer to https://github.com/jiwoogit/StyleID for how to compute ArtFID. :p
"""
@InProceedings{Chung_2024_CVPR,
    author    = {Chung, Jiwoo and Hyun, Sangeek and Heo, Jae-Pil},
    title     = {Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8795-8805}
}
"""
import os
import shutil
from natsort import natsorted

MODEL = "stylos"  # Change the model name to yours
SRC = f"/projects/bfcb/hliu15/3d_style_sota/{MODEL}"  # Change the src dir to yours
DST = f"/projects/bfcb/hliu15/3d_style_sota_artfid/{MODEL}"  # Change the dst dir to yours

SCENES = ["garden", "M60", "train", "truck"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_style_folder(name):
    """Check if folder's last two chars are valid style ID 00–49."""
    if len(name) < 2:
        return False
    suf = name[-2:]
    return suf.isdigit() and 0 <= int(suf) <= 49

def main():
    for scene in SCENES:
        print(f"\n=== Processing scene: {scene} ===")

        scene_src = os.path.join(SRC, scene)
        scene_dst = os.path.join(DST, scene)
        ensure_dir(scene_dst)

        style_folders = [
            f for f in os.listdir(scene_src)
            if os.path.isdir(os.path.join(scene_src, f)) and is_style_folder(f)
        ]

        style_folders = natsorted(style_folders)
        print(f"  Style folders found: {style_folders}")

        if not style_folders:
            print("No style dirs found, skip.")
            continue

        for style_name in style_folders:
            style_id = style_name[-2:]   # like 00, 01, ...
            style_path = os.path.join(scene_src, style_name)

            imgs = [
                f for f in os.listdir(style_path)
                if f.lower().endswith(".png") or f.lower().endswith(".jpg")
            ]

            print(f"  Style {style_id}: {len(imgs)} images")

            for img in imgs:
                src_img = os.path.join(style_path, img)
                dst_img = os.path.join(scene_dst, f"{style_id}_{img}")

                shutil.copy(src_img, dst_img)

        print(f"Scene {scene} completed.")

    print("\nAll scenes processed successfully.")

if __name__ == "__main__":
    main()

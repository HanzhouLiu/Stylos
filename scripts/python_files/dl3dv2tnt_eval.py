import sys
sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
import lpips
import csv
from torchvision import models, transforms
from torch import nn

from raft_core.raft import RAFT
from raft_core.utils.utils import InputPadder
import raft_core.softsplat as softsplat


##########################################################
# Backwarp function for RAFT-based flow warping
##########################################################

backwarp_tenGrid = {}

def backwarp(tenIn, tenFlow):
    """
    Backwarp an input tensor according to a flow field. This is used to 
    compute temporal consistency errors by warping one frame toward another.
    """

    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]),
                                1.0 - (1.0 / tenFlow.shape[3]),
                                tenFlow.shape[3]).view(1,1,1,-1).repeat(1,1,tenFlow.shape[2],1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]),
                                1.0 - (1.0 / tenFlow.shape[2]),
                                tenFlow.shape[2]).view(1,1,-1,1).repeat(1,1,1,tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat([
        tenFlow[:, 0:1] / ((tenIn.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2] / ((tenIn.shape[2] - 1.0) / 2.0)
    ], 1)

    return torch.nn.functional.grid_sample(
        input=tenIn,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0,2,3,1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )


##########################################################
# Utility functions
##########################################################

DEVICE = 'cuda'

def load_image(imfile):
    """
    Load a single image (RGB) as a torch tensor and move it to GPU.
    """
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2,0,1).float()
    return img[None].to(DEVICE)


def load_images_from_directory(directory):
    """
    Load all images from a folder. Automatically checks for direct images 
    and an 'images' subfolder if needed. Performs a natural numeric sort.
    """

    extensions = ['*.png','*.PNG','*.jpg','*.JPG','*.jpeg','*.JPEG']
    images = []

    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))

    if len(images) == 0:
        subdir = os.path.join(directory, 'images')
        for ext in extensions:
            images.extend(glob.glob(os.path.join(subdir, ext)))

    import re
    def natural_sort_key(filename):
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[-1]) if numbers else filename

    return sorted(images, key=natural_sort_key)


##########################################################
# Flow-based evaluation functions
##########################################################

def evaluate_frame_pair(model, image1, image2, loss_fn_vgg, alpha=10):
    """
    Compute LPIPS and RMSE between two frames using RAFT optical flow
    and soft splatting for forward warping.
    """

    padder = InputPadder(image1.shape)
    image1_padded, image2_padded = padder.pad(image1, image2)

    _, flow_up = model(image1_padded, image2_padded, iters=32, test_mode=True)

    tenMetric = torch.nn.functional.l1_loss(
        input=image1_padded,
        target=backwarp(image2_padded, flow_up),
        reduction='none'
    ).mean([1], True)

    tenSoftmax = softsplat.softsplat(
        tenIn=image1_padded,
        tenFlow=flow_up,
        tenMetric=(-alpha * tenMetric).clip(-alpha, alpha),
        strMode='soft'
    )

    mask = torch.where(tenSoftmax.sum(1, keepdim=True)==0., 0., 1.)
    lpips_score = loss_fn_vgg(tenSoftmax, image2_padded * mask)

    rmse_score = torch.sqrt(torch.nn.functional.mse_loss(
        tenSoftmax / 255., image2_padded * mask / 255., reduction='mean'
    ))

    return lpips_score.item(), rmse_score.item()


def evaluate_short_range(model, image_paths, loss_fn_vgg):
    """
    Evaluate consistency between consecutive frames.
    """
    lpips_vals, rmse_vals = [], []

    for i in range(len(image_paths)-1):
        img1 = load_image(image_paths[i])
        img2 = load_image(image_paths[i+1])
        lp, rm = evaluate_frame_pair(model, img1, img2, loss_fn_vgg)
        lpips_vals.append(lp)
        rmse_vals.append(rm)

    return np.mean(lpips_vals), np.mean(rmse_vals)


def evaluate_long_range(model, image_paths, loss_fn_vgg, gap=7):
    """
    Evaluate long-range consistency: frame i vs frame i-gap.
    """
    lpips_vals, rmse_vals = [], []

    for i in range(gap, len(image_paths)):
        img1 = load_image(image_paths[i-gap])
        img2 = load_image(image_paths[i])
        lp, rm = evaluate_frame_pair(model, img1, img2, loss_fn_vgg)
        lpips_vals.append(lp)
        rmse_vals.append(rm)

    return np.mean(lpips_vals), np.mean(rmse_vals)


##########################################################
# Evaluation per scene: ArtScore
##########################################################

def evaluate_artscore_for_scene(scene_path, artscore_model, artscore_transform):
    """
    Evaluate ArtScore for each style in the scene.
    Returns a list of tuples: (style_name, num_images, artscore_mean)
    """

    styles_dir = scene_path
    results = []

    with torch.no_grad():
        for style in sorted(os.listdir(styles_dir)):
            style_dir = os.path.join(styles_dir, style)
            if not os.path.isdir(style_dir):
                continue

            img_paths = load_images_from_directory(style_dir)
            if len(img_paths) == 0:
                continue

            scores = []
            for img_path in img_paths:
                img = Image.open(img_path).convert("RGB")
                inp = artscore_transform(img).unsqueeze(0).to(DEVICE)
                score = artscore_model(inp)[0].item()
                scores.append(score)

            results.append((style, len(img_paths), np.mean(scores)))

    return results


##########################################################
# Evaluation per scene: Flow-based consistency
##########################################################

def evaluate_consistency_for_scene(scene_path, raft_model, loss_fn_vgg, long_gap=7):
    """
    Evaluate short-range and long-range temporal consistency for each style.
    Returns rows of: (style, num_images, lp_s, rm_s, lp_l, rm_l)
    """

    styles_dir = scene_path
    results = []

    with torch.no_grad():
        for style in sorted(os.listdir(styles_dir)):
            style_dir = os.path.join(styles_dir, style)
            if not os.path.isdir(style_dir):
                continue

            paths = load_images_from_directory(style_dir)
            if len(paths) == 0:
                continue

            lp_s = rm_s = lp_l = rm_l = None

            if len(paths) > 1:
                lp_s, rm_s = evaluate_short_range(raft_model, paths, loss_fn_vgg)

            if len(paths) > long_gap:
                lp_l, rm_l = evaluate_long_range(raft_model, paths, loss_fn_vgg, gap=long_gap)

            results.append((style, len(paths), lp_s, rm_s, lp_l, rm_l))

    return results


##########################################################
# Main evaluation entry
##########################################################

def main_eval(args):

    # Load RAFT model
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module.to(DEVICE).eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # Load ArtScore model
    artscore_ckpt = torch.load(args.artscore_model)
    artscore_model = models.resnet50()
    artscore_model.fc = nn.Sequential(
        nn.Linear(2048,1000),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1000,1),
    )
    artscore_model.load_state_dict(artscore_ckpt)
    artscore_model = artscore_model.to(DEVICE).eval()

    artscore_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Scene path
    scene_path = os.path.join(args.base_dir, args.scene)
    print(f"Evaluating scene: {args.scene}")
    print(f"Scene path: {scene_path}")

    # Evaluate ArtScore and save CSV
    artscore_results = evaluate_artscore_for_scene(
        scene_path, artscore_model, artscore_transform
    )
    artscore_csv = os.path.join(scene_path, "reb_artscore.csv")
    with open(artscore_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Style","NumImages","ArtScore"])
        
        for row in artscore_results:
            writer.writerow(row)

        # Compute and write Overall_Avg
        if len(artscore_results) > 0:
            num_total = sum(r[1] for r in artscore_results)
            avg_artscore = np.mean([r[2] for r in artscore_results])
            writer.writerow(["Overall_Avg", num_total, avg_artscore])

    print(f"ArtScore results saved to: {artscore_csv}")

    # Evaluate consistency and save CSV
    consistency_results = evaluate_consistency_for_scene(
        scene_path, raft_model, loss_fn_vgg, long_gap=args.long_gap
    )
    cons_csv = os.path.join(scene_path, "reb_consistency.csv")
    with open(cons_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Style","NumImages","LPIPS_short","RMSE_short","LPIPS_long","RMSE_long"])
        
        for row in consistency_results:
            writer.writerow(row)

        # Compute Overall_Avg
        if len(consistency_results) > 0:
            num_total = sum(r[1] for r in consistency_results)

            # Extract valid values only (avoid None)
            lp_s_vals = [r[2] for r in consistency_results if r[2] is not None]
            rm_s_vals = [r[3] for r in consistency_results if r[3] is not None]
            lp_l_vals = [r[4] for r in consistency_results if r[4] is not None]
            rm_l_vals = [r[5] for r in consistency_results if r[5] is not None]

            avg_lp_s = np.mean(lp_s_vals) if lp_s_vals else None
            avg_rm_s = np.mean(rm_s_vals) if rm_s_vals else None
            avg_lp_l = np.mean(lp_l_vals) if lp_l_vals else None
            avg_rm_l = np.mean(rm_l_vals) if rm_l_vals else None

            writer.writerow([
                "Overall_Avg",
                num_total,
                avg_lp_s,
                avg_rm_s,
                avg_lp_l,
                avg_rm_l
            ])

    print(f"Consistency results saved to: {cons_csv}")


##########################################################
# Argument parsing
##########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='checkpoints/raft-things.pth',
                        help="Path to RAFT checkpoint.")
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')

    parser.add_argument('--base_dir', type=str, required=True,
                        help="Base directory containing scene folders.")
    parser.add_argument('--scene', type=str, required=True,
                        help="Scene name (e.g. garden, m60, train, truck).")

    parser.add_argument('--mode', choices=['short','long','both'], default='both')
    parser.add_argument('--long_gap', type=int, default=7,
                        help="Gap for long-range consistency evaluation.")

    parser.add_argument('--artscore_model', type=str, required=True,
                        help="Path to ArtScore model.")

    args = parser.parse_args()
    main_eval(args)

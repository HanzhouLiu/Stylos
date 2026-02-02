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
# Backwarp Helper (for flow-based metrics)
##########################################################
backwarp_tenGrid = {}

def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]),
                                1.0 - (1.0 / tenFlow.shape[3]),
                                tenFlow.shape[3]).view(1,1,1,-1).repeat(1,1,tenFlow.shape[2],1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]),
                                1.0 - (1.0 / tenFlow.shape[2]),
                                tenFlow.shape[2]).view(1,1,-1,1).repeat(1,1,1,tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer],1).cuda()

    tenFlow = torch.cat([
        tenFlow[:,0:1,:,:]/((tenIn.shape[3]-1.0)/2.0),
        tenFlow[:,1:2,:,:]/((tenIn.shape[2]-1.0)/2.0)
    ],1)

    return torch.nn.functional.grid_sample(
        input=tenIn,
        grid=(backwarp_tenGrid[str(tenFlow.shape)]+tenFlow).permute(0,2,3,1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

##########################################################
# Utilities
##########################################################
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2,0,1).float()
    return img[None].to(DEVICE)

def load_images_from_directory(directory):
    """
    Load all images from a given directory, automatically checking for a possible 'images' subfolder.
    Supports multiple common extensions and both lowercase and uppercase filenames.
    Returns a naturally sorted list of image paths.
    """
    # Supported image extensions (case-insensitive)
    extensions = ['*.png','*.PNG','*.jpg','*.JPG','*.jpeg','*.JPEG']
    images = []

    # First, check the directory itself
    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))

    # If no images are found, check the 'images' subdirectory
    if len(images) == 0:
        subdir = os.path.join(directory, 'images')
        for ext in extensions:
            images.extend(glob.glob(os.path.join(subdir, ext)))

    # Natural sort function to sort filenames by number
    import re
    def natural_sort_key(filename):
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[-1]) if numbers else filename

    # Return sorted list of image paths
    return sorted(images, key=natural_sort_key)


##########################################################
# Flow-based Evaluation Functions
##########################################################
def evaluate_frame_pair(model, image1, image2, loss_fn_vgg, alpha=10):
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
        tenMetric=(-alpha*tenMetric).clip(-alpha,alpha),
        strMode='soft'
    )

    mask = torch.where(tenSoftmax.sum(1, keepdim=True)==0.,0.,1.)
    lpips_score = loss_fn_vgg(tenSoftmax, image2_padded*mask)
    rmse_score = torch.sqrt(torch.nn.functional.mse_loss(
        tenSoftmax/255., image2_padded*mask/255., reduction='mean'
    ))

    return lpips_score.item(), rmse_score.item()

def evaluate_short_range(model, image_paths, loss_fn_vgg):
    lpips_vals, rmse_vals = [], []
    for i in range(len(image_paths)-1):
        img1, img2 = load_image(image_paths[i]), load_image(image_paths[i+1])
        lp, rm = evaluate_frame_pair(model, img1, img2, loss_fn_vgg)
        lpips_vals.append(lp); rmse_vals.append(rm)
    return np.mean(lpips_vals), np.mean(rmse_vals)

def evaluate_long_range(model, image_paths, loss_fn_vgg, gap=7):
    lpips_vals, rmse_vals = [], []
    for i in range(gap, len(image_paths)):
        img1, img2 = load_image(image_paths[i-gap]), load_image(image_paths[i])
        lp, rm = evaluate_frame_pair(model, img1, img2, loss_fn_vgg)
        lpips_vals.append(lp); rmse_vals.append(rm)
    return np.mean(lpips_vals), np.mean(rmse_vals)

##########################################################
# Main Evaluation Loop
##########################################################
def main_eval(args):
    # --------------------------
    # Initialize flow-based model
    # --------------------------
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module.to(DEVICE).eval()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # --------------------------
    # Initialize ArtScore model
    # --------------------------
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

    # --------------------------
    # Results storage
    # --------------------------
    scene_results = []

    with torch.no_grad():
        for scene_dir in sorted(os.listdir(args.base_dir)):
            scene_path = os.path.join(args.base_dir, scene_dir, "styles")
            if not os.path.isdir(scene_path):
                continue

            print(f"\n[Scene: {scene_dir}]")

            lp_s_list, rm_s_list, lp_l_list, rm_l_list = [], [], [], []
            scene_artscore_list = []

            for style_dir in sorted(os.listdir(scene_path)):
                style_path = os.path.join(scene_path, style_dir)
                if not os.path.isdir(style_path):
                    continue

                paths = load_images_from_directory(style_path)
                if len(paths) < 1:
                    continue

                # --------------------------
                # 1. Compute ArtScore first
                # --------------------------
                art_scores = []
                for img_path in paths:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = artscore_transform(image).unsqueeze(0).to(DEVICE)
                    score = artscore_model(input_tensor)[0].item()
                    art_scores.append(score)
                style_artscore = np.mean(art_scores)
                scene_artscore_list.append(style_artscore)

                # --------------------------
                # 2. Compute flow-based metrics
                # --------------------------
                if args.mode in ["both","short"] and len(paths) > 1:
                    lp_s, rm_s = evaluate_short_range(raft_model, paths, loss_fn_vgg)
                    lp_s_list.append(lp_s); rm_s_list.append(rm_s)

                if args.mode in ["both","long"] and len(paths) > args.long_gap:
                    lp_l, rm_l = evaluate_long_range(raft_model, paths, loss_fn_vgg, gap=args.long_gap)
                    lp_l_list.append(lp_l); rm_l_list.append(rm_l)

            # --------------------------
            # Save scene-level metrics
            # --------------------------
            scene_results.append((
                scene_dir,
                np.mean(lp_s_list) if lp_s_list else None,
                np.mean(rm_s_list) if rm_s_list else None,
                np.mean(lp_l_list) if lp_l_list else None,
                np.mean(rm_l_list) if rm_l_list else None,
                np.mean(scene_artscore_list) if scene_artscore_list else None
            ))

            print(f"  Scene ArtScore Avg: {np.mean(scene_artscore_list):.4f}" if scene_artscore_list else "")

    # --------------------------
    # Write results to CSV
    # --------------------------
    csv_path = os.path.join(args.base_dir, "eval_results_final.csv")
    header = ["Scene","LPIPS_short","RMSE_short","LPIPS_long","RMSE_long","ArtScore"]
    with open(csv_path,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in scene_results:
            writer.writerow(row)

        # Overall averages
        if scene_results:
            avg_lp_s = np.nanmean([r[1] for r in scene_results])
            avg_rm_s = np.nanmean([r[2] for r in scene_results])
            avg_lp_l = np.nanmean([r[3] for r in scene_results])
            avg_rm_l = np.nanmean([r[4] for r in scene_results])
            avg_art = np.nanmean([r[5] for r in scene_results])
            writer.writerow(["Overall_Avg", avg_lp_s, avg_rm_s, avg_lp_l, avg_rm_l, avg_art])

    print(f"\n[INFO] Results saved to {csv_path}")

##########################################################
# Argument Parsing
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/raft-things.pth', help="RAFT checkpoint path")
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory containing all scenes")
    parser.add_argument('--mode', choices=['short','long','both'], default='both')
    parser.add_argument('--long_gap', type=int, default=7, help="Frame gap for long-range evaluation")
    parser.add_argument('--artscore_model', type=str, required=True, help="ArtScore model checkpoint path")
    args = parser.parse_args()
    main_eval(args)

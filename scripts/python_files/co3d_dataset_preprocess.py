#!/usr/bin/env python3
"""
Script to adapt existing CO3D processed data to match DataLoader expectations.
This script PRESERVES all existing files and only ADDS necessary files.
"""

import argparse
import json
import gzip
import os
import os.path as osp
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import cv2
from PIL import Image
import re

import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dust3r.datasets.utils.cropping import rescale_image_depthmap, crop_image_depthmap


CATEGORIES = [
    "chair", "couch", "bench", "bottle", "cup", "donut",
    "car", "bicycle", "motorcycle", "skateboard", 
    "apple", "banana", "pizza", "cake",
    "cellphone", "laptop", "handbag",
    "teddybear", "umbrella", "toaster"
]

#CATEGORIES = [
#    "donut"
#]

def get_parser():
    parser = argparse.ArgumentParser(description="Adapt CO3D data for DataLoader")
    parser.add_argument("--co3d_root", type=str, required=True, help="Root directory of CO3D data")
    parser.add_argument("--categories", type=str, nargs="+", default=None, 
                       help=f"Categories to process (default: all predefined categories: {CATEGORIES})")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without actually doing it")
    return parser


def find_categories(co3d_root):
    """Find all category directories that contain the expected structure"""
    categories = []
    for item in os.listdir(co3d_root):
        category_path = osp.join(co3d_root, item)
        if osp.isdir(category_path):
            # Check if it has the expected annotation files
            if (osp.exists(osp.join(category_path, "frame_annotations.jgz")) and 
                osp.exists(osp.join(category_path, "sequence_annotations.jgz"))):
                categories.append(item)
    return sorted(categories)


def find_scenes_in_category(category_path):
    """Find all scene directories in a category"""
    scenes = []
    for item in os.listdir(category_path):
        scene_path = osp.join(category_path, item)
        if osp.isdir(scene_path) and item not in ["frame_annotations.jgz", "sequence_annotations.jgz"]:
            # Check if it has the expected subdirectories
            if (osp.exists(osp.join(scene_path, "images")) and 
                osp.exists(osp.join(scene_path, "depths")) and
                osp.exists(osp.join(scene_path, "masks"))):
                scenes.append(item)
    return sorted(scenes)


def load_annotations(category_path):
    """Load frame and sequence annotations"""
    frame_file = osp.join(category_path, "frame_annotations.jgz")
    sequence_file = osp.join(category_path, "sequence_annotations.jgz")
    
    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())
    
    # Process frame data by sequence
    frame_data_by_seq = defaultdict(dict)
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_by_seq[sequence_name][f_data["frame_number"]] = f_data
    
    return frame_data_by_seq, sequence_data


def extract_frame_number_from_filename(filename):
    """Extract frame number from various filename formats"""
    # Try different patterns
    patterns = [
        r"frame(\d+)",           # frame123.jpg
        r"frame_(\d+)",          # frame_123.jpg  
        r"(\d+)",                # 123.jpg
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return None


def get_existing_frames(scene_path):
    """Get list of existing frame files and their numbers"""
    images_path = osp.join(scene_path, "images")
    if not osp.exists(images_path):
        return []
    
    frames = []
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame_num = extract_frame_number_from_filename(filename)
            if frame_num is not None:
                frames.append((frame_num, filename))
    
    return sorted(frames)


def normalize_filename(frame_num):
    """Generate normalized filename format"""
    return f"frame{frame_num:06d}"


def ensure_normalized_filenames(scene_path, frames, dry_run=False):
    """Ensure files have normalized names (create symlinks if needed)"""
    images_path = osp.join(scene_path, "images")
    depths_path = osp.join(scene_path, "depths") 
    masks_path = osp.join(scene_path, "masks")
    
    normalized_frames = []
    
    for frame_num, original_filename in frames:
        normalized_base = normalize_filename(frame_num)
        
        # Handle image files
        original_img = osp.join(images_path, original_filename)
        normalized_img = osp.join(images_path, f"{normalized_base}.jpg")
        
        if original_img != normalized_img and not osp.exists(normalized_img):
            if not dry_run:
                if osp.exists(original_img):
                    os.symlink(osp.basename(original_img), normalized_img)
            print(f"  Link: {original_img} -> {normalized_img}")
        
        # Handle depth files (try multiple possible extensions)
        depth_extensions = [".png.geometric.png", ".jpg.geometric.png", ".png", ".jpg"]
        for ext in depth_extensions:
            original_depth = osp.join(depths_path, original_filename.replace(".jpg", ext).replace(".png", ext))
            if osp.exists(original_depth):
                normalized_depth = osp.join(depths_path, f"{normalized_base}.jpg.geometric.png")
                if original_depth != normalized_depth and not osp.exists(normalized_depth):
                    if not dry_run:
                        os.symlink(osp.basename(original_depth), normalized_depth)
                    print(f"  Link: {original_depth} -> {normalized_depth}")
                break
        
        # Handle mask files
        mask_extensions = [".png", ".jpg"]
        for ext in mask_extensions:
            original_mask = osp.join(masks_path, original_filename.replace(".jpg", ext).replace(".jpeg", ext))
            if osp.exists(original_mask):
                normalized_mask = osp.join(masks_path, f"{normalized_base}.png")
                if original_mask != normalized_mask and not osp.exists(normalized_mask):
                    if not dry_run:
                        os.symlink(osp.basename(original_mask), normalized_mask)
                    print(f"  Link: {original_mask} -> {normalized_mask}")
                break
        
        normalized_frames.append(frame_num)
    
    return normalized_frames


def analyze_scene_frame_coverage(category_path, scenes, frame_data_by_seq):
    """Analyze frame coverage across all scenes"""
    print(f"\n  === Scene Frame Coverage Analysis ===")
    
    coverage_stats = {
        'valid_direct': [],        # Direct mapping, equal counts
        'valid_off_by_one': [],    # Off-by-one mapping, equal counts  
        'invalid_count_mismatch': [], # Different image/annotation counts
        'invalid_no_pattern': [],  # Equal counts but no valid mapping pattern
        'no_annotations': []       # No annotation data found
    }
    
    for scene in scenes[:10]:  # Check first 10 scenes
        scene_path = osp.join(category_path, scene)
        existing_frames = get_existing_frames(scene_path)
        
        if not existing_frames:
            continue
            
        image_frame_nums = set([frame_num for frame_num, _ in existing_frames])
        
        if scene in frame_data_by_seq:
            annotation_frame_nums = set(frame_data_by_seq[scene].keys())
            
            # Check count equality first
            if len(image_frame_nums) != len(annotation_frame_nums):
                coverage_stats['invalid_count_mismatch'].append(scene)
                print(f"    {scene}: INVALID - Count mismatch (Images={len(image_frame_nums)}, Annotations={len(annotation_frame_nums)})")
                continue
            
            # Equal counts - check mapping patterns
            min_img, max_img = min(image_frame_nums), max(image_frame_nums)
            min_ann, max_ann = min(annotation_frame_nums), max(annotation_frame_nums)
            
            # Check for off-by-one pattern
            is_off_by_one = (min_img == 1 and min_ann == 0 and max_img == max_ann + 1)
            
            if is_off_by_one:
                # Verify all frames can be mapped
                can_map_all = all((img-1) in annotation_frame_nums for img in image_frame_nums)
                if can_map_all:
                    coverage_stats['valid_off_by_one'].append(scene)
                    print(f"    {scene}: VALID off-by-one (Images {min_img}-{max_img} → Annotations {min_ann}-{max_ann})")
                else:
                    coverage_stats['invalid_no_pattern'].append(scene)
                    print(f"    {scene}: INVALID - Off-by-one pattern but incomplete mapping")
            
            # Check for direct mapping  
            elif image_frame_nums == annotation_frame_nums:
                coverage_stats['valid_direct'].append(scene)
                print(f"    {scene}: VALID direct mapping ({min_img}-{max_img})")
            
            else:
                coverage_stats['invalid_no_pattern'].append(scene)
                print(f"    {scene}: INVALID - No valid mapping pattern")
                print(f"      Images: {min_img}-{max_img}, Annotations: {min_ann}-{max_ann}")
                
        else:
            coverage_stats['no_annotations'].append(scene)
            print(f"    {scene}: No annotation data found")
    
    print(f"  Valid (direct mapping): {len(coverage_stats['valid_direct'])}")
    print(f"  Valid (off-by-one): {len(coverage_stats['valid_off_by_one'])}")  
    print(f"  Invalid (count mismatch): {len(coverage_stats['invalid_count_mismatch'])}")
    print(f"  Invalid (no valid pattern): {len(coverage_stats['invalid_no_pattern'])}")
    print(f"  No annotations: {len(coverage_stats['no_annotations'])}")
    
    return coverage_stats


def filter_frames_with_annotations(valid_frames, frame_data_by_seq, scene_name):
    """Filter frames to only include those that have annotation data"""
    if scene_name not in frame_data_by_seq:
        print(f"      Warning: No sequence data found for {scene_name}")
        return None
    
    sequence_frame_data = frame_data_by_seq[scene_name]
    available_annotation_frames = set(sequence_frame_data.keys())
    valid_image_frames = set(valid_frames)
    
    # Check if counts match exactly
    if len(valid_image_frames) != len(available_annotation_frames):
        print(f"      INVALID SCENE: Image count ({len(valid_image_frames)}) != Annotation count ({len(available_annotation_frames)})")
        print(f"      Skipping scene due to data mismatch")
        return None
    
    # Try direct match first
    direct_intersection = available_annotation_frames & valid_image_frames
    
    # Check for off-by-one pattern: images 1-N, annotations 0-(N-1)
    min_img, max_img = min(valid_image_frames), max(valid_image_frames)
    min_ann, max_ann = min(available_annotation_frames), max(available_annotation_frames)
    
    is_off_by_one_pattern = (min_img == 1 and min_ann == 0 and max_img == max_ann + 1)
    
    if is_off_by_one_pattern:
        print(f"      Detected off-by-one pattern: Images {min_img}-{max_img}, Annotations {min_ann}-{max_ann}")
        
        # Apply off-by-one mapping: image N -> annotation N-1
        frames_with_both = []
        frame_mapping = {}
        
        for img_frame in sorted(valid_image_frames):
            corresponding_ann_frame = img_frame - 1
            if corresponding_ann_frame in available_annotation_frames:
                frames_with_both.append(img_frame)
                frame_mapping[img_frame] = corresponding_ann_frame
        
        print(f"      Off-by-one mapping: {len(frames_with_both)} frames mapped")
        
        # Must be perfect mapping for valid scene
        if len(frames_with_both) == len(available_annotation_frames) == len(valid_image_frames):
            print(f"      Perfect! All frames mapped successfully")
            return frames_with_both, frame_mapping
        else:
            print(f"      INVALID SCENE: Cannot achieve perfect off-by-one mapping")
            return None
    
    elif len(direct_intersection) == len(available_annotation_frames) == len(valid_image_frames):
        # Direct matching worked perfectly
        frames_with_both = sorted(list(direct_intersection))
        frame_mapping = {f: f for f in frames_with_both}
        
        print(f"      Using {len(frames_with_both)} frames with direct mapping (valid scene)")
        return frames_with_both, frame_mapping
    
    else:
        print(f"      INVALID SCENE: No valid mapping pattern found")
        print(f"      Direct intersection: {len(direct_intersection)}, Images: {len(valid_image_frames)}, Annotations: {len(available_annotation_frames)}")
        return None


def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    """Convert NDC camera parameters to pinhole format"""
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    """Convert camera parameters to OpenCV format"""
    import torch
    
    R = torch.from_numpy(np.array(R))[None, :, :]
    T = torch.from_numpy(np.array(T))[None, :]
    focal = torch.from_numpy(np.array(focal))[None, :]
    p0 = torch.from_numpy(np.array(p0))[None, :]
    image_size = torch.from_numpy(np.array(image_size))[None, :]

    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    
    return R[0].numpy(), tvec[0].numpy(), camera_matrix[0].numpy()


def generate_npz_files(scene_path, scene_name, valid_frames, frame_data_by_seq, frame_mapping, dry_run=False, img_size=512, normalize_to_first=True):
    """Generate .npz files with corrected camera parameters for each frame.
    If normalize_to_first=True, all poses are expressed relative to the first frame (first frame = identity).
    """

    images_path = osp.join(scene_path, "images")
    depths_path = osp.join(scene_path, "depths")
    masks_path = osp.join(scene_path, "masks")

    if scene_name not in frame_data_by_seq:
        print(f"  Warning: No frame data found for sequence {scene_name}")
        return

    sequence_frame_data = frame_data_by_seq[scene_name]
    first_pose_ref = None  # 保存第一帧的位姿（c2w）

    for image_frame_num in valid_frames:
        npz_path = osp.join(images_path, f"frame{image_frame_num:06d}.npz")
        if osp.exists(npz_path):
            continue  # Skip if already exists

        annotation_frame_num = frame_mapping.get(image_frame_num)
        if annotation_frame_num is None:
            print(f"      Warning: No mapping for image frame {image_frame_num}")
            continue

        frame_data = sequence_frame_data.get(annotation_frame_num)
        if frame_data is None:
            print(f"      Warning: No annotation for frame {annotation_frame_num} (image {image_frame_num})")
            continue

        # 文件路径
        img_path = osp.join(images_path, f"frame{image_frame_num:06d}.jpg")
        mask_path = osp.join(masks_path, f"frame{image_frame_num:06d}.png")
        depth_path = osp.join(depths_path, f"frame{image_frame_num:06d}.jpg.geometric.png")

        if not (osp.exists(img_path) and osp.exists(mask_path) and osp.exists(depth_path)):
            print(f"  Warning: Missing files for frame {image_frame_num}")
            continue

        try:
            # === 加载数据 ===
            input_rgb_image = Image.open(img_path).convert("RGB")
            input_mask = cv2.imread(mask_path, -1).astype(np.float32) / 255.0
            input_depthmap = cv2.imread(depth_path, -1).astype(np.float32)

            # 组合深度和mask
            depth_mask = np.stack((input_depthmap, input_mask), axis=-1)
            H, W = input_depthmap.shape

            # === 相机参数 ===
            focal_length = frame_data["viewpoint"]["focal_length"]
            principal_point = frame_data["viewpoint"]["principal_point"]
            R_in = np.array(frame_data["viewpoint"]["R"])
            T_in = np.array(frame_data["viewpoint"]["T"])
            image_size = [H, W]

            R_w2c, tvec, camera_intrinsics = opencv_from_cameras_projection(
                R_in, T_in, focal_length, principal_point, image_size
            )
            camera_intrinsics = (
                camera_intrinsics if not hasattr(camera_intrinsics, "numpy") else camera_intrinsics.numpy()
            )

            # === 裁剪 ===
            cx, cy = camera_intrinsics[:2, 2].round().astype(int)
            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)
            l, t, r, b = cx - min_margin_x, cy - min_margin_y, cx + min_margin_x, cy + min_margin_y
            crop_bbox = (l, t, r, b)

            input_rgb_image, depth_mask, input_camera_intrinsics = crop_image_depthmap(
                input_rgb_image, depth_mask, camera_intrinsics, crop_bbox
            )

            # === 缩放 ===
            input_depthmap = depth_mask[:, :, 0]
            H, W = input_depthmap.shape
            scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
            if max(output_resolution) < img_size:
                scale_final = (img_size / max(H, W)) + 1e-8
                output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

            input_rgb_image, depth_mask, input_camera_intrinsics = rescale_image_depthmap(
                input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution
            )

            # === 提取缩放后的数据 ===
            input_depthmap = depth_mask[:, :, 0]
            input_mask = depth_mask[:, :, 1]
            maximum_depth = float(np.max(input_depthmap))

            # === 相机位姿 (c2w) ===
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = R_w2c
            camera_pose[:3, 3] = tvec
            camera_pose = np.linalg.inv(camera_pose)

            # === 归一化到第一帧 ===
            if normalize_to_first:
                if first_pose_ref is None:
                    first_pose_ref = camera_pose.copy()
                camera_pose = np.linalg.inv(first_pose_ref) @ camera_pose

            # === 保存 ===
            if not dry_run:
                # RGB
                input_rgb_image.save(img_path)

                # Depth (uint16, 0~65535)
                scaled_depth_map = (input_depthmap / maximum_depth * 65535).astype(np.uint16)
                cv2.imwrite(depth_path, scaled_depth_map)

                # Mask (uint8, 0~255)
                cv2.imwrite(mask_path, (input_mask * 255).astype(np.uint8))

                # Meta
                np.savez(
                    npz_path,
                    camera_intrinsics=input_camera_intrinsics,
                    camera_pose=camera_pose,
                    maximum_depth=maximum_depth,
                )

            print(f"  Generated: {npz_path}")

        except Exception as e:
            print(f"  Error processing frame {image_frame_num}: {e}")


def generate_analysis_report(category_path, category_name, coverage_stats, dry_run=False):
    """Generate a detailed analysis report for the category"""
    report_path = osp.join(category_path, "scene_analysis_report.json")
    
    if osp.exists(report_path):
        print(f"  Analysis report already exists, skipping...")
        return
    
    # Create detailed report
    report_data = {
        "category": category_name,
        "analysis_summary": {
            "valid_direct_mapping": len(coverage_stats['valid_direct']),
            "valid_off_by_one_mapping": len(coverage_stats['valid_off_by_one']),
            "invalid_count_mismatch": len(coverage_stats['invalid_count_mismatch']),
            "invalid_no_pattern": len(coverage_stats['invalid_no_pattern']),
            "no_annotations": len(coverage_stats['no_annotations']),
            "total_scenes_analyzed": sum(len(v) for v in coverage_stats.values()),
            "total_valid_scenes": len(coverage_stats['valid_direct']) + len(coverage_stats['valid_off_by_one'])
        },
        "scene_details": {
            "valid_scenes": {
                "direct_mapping": coverage_stats['valid_direct'],
                "off_by_one_mapping": coverage_stats['valid_off_by_one']
            },
            "invalid_scenes": {
                "count_mismatch": coverage_stats['invalid_count_mismatch'],
                "no_valid_pattern": coverage_stats['invalid_no_pattern'],
                "no_annotations": coverage_stats['no_annotations']
            }
        },
        "notes": {
            "direct_mapping": "Scene has identical frame numbers for images and annotations",
            "off_by_one_mapping": "Scene uses 1-based image indexing (1-N) with 0-based annotation indexing (0-N-1)",
            "count_mismatch": "Number of images != number of annotations",
            "no_valid_pattern": "Equal counts but neither direct nor off-by-one mapping works",
            "no_annotations": "No annotation data found for this scene"
        }
    }
    
    if not dry_run:
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    print(f"  Generated analysis report: scene_analysis_report.json")
    print(f"    Total scenes: {report_data['analysis_summary']['total_scenes_analyzed']}")
    print(f"    Valid scenes: {report_data['analysis_summary']['total_valid_scenes']}")
    print(f"    Invalid scenes: {report_data['analysis_summary']['total_scenes_analyzed'] - report_data['analysis_summary']['total_valid_scenes']}")


def generate_valid_seq_json(category_path, scenes_data, dry_run=False):
    """Generate valid_seq.json file"""
    valid_seq_path = osp.join(category_path, "valid_seq.json")
    
    if osp.exists(valid_seq_path):
        print(f"  valid_seq.json already exists, skipping...")
        return
    
    valid_seq_data = {}
    for scene_name, frame_list in scenes_data.items():
        valid_seq_data[scene_name] = sorted(frame_list)
    
    if not dry_run:
        with open(valid_seq_path, 'w') as f:
            json.dump(valid_seq_data, f, indent=2)
    
    print(f"  Generated: valid_seq.json with {len(valid_seq_data)} scenes")


def process_category(co3d_root, category, dry_run=False):
    """Process a single category"""
    print(f"\nProcessing category: {category}")
    
    category_path = osp.join(co3d_root, category)
    
    # Load annotations
    try:
        frame_data_by_seq, sequence_data = load_annotations(category_path)
        print(f"  Loaded annotations for {len(frame_data_by_seq)} sequences")
    except Exception as e:
        print(f"  Error loading annotations: {e}")
        return
    
    # Find scenes
    scenes = find_scenes_in_category(category_path)
    print(f"  Found {len(scenes)} scenes")
    
    # Analyze frame coverage across scenes
    coverage_stats = analyze_scene_frame_coverage(category_path, scenes, frame_data_by_seq)
    
    scenes_data = {}
    
    for scene in tqdm(scenes, desc=f"Processing {category} scenes"):
        scene_path = osp.join(category_path, scene)
        print(f"    Processing scene: {scene}")
        
        # Get existing frames
        existing_frames = get_existing_frames(scene_path)
        if not existing_frames:
            print(f"      No valid frames found in {scene}")
            continue
        
        print(f"      Found {len(existing_frames)} frames")
        
        # Filter frames to only include those with annotations
        all_frame_numbers = [frame_num for frame_num, _ in existing_frames]
        result = filter_frames_with_annotations(
            all_frame_numbers, frame_data_by_seq, scene)
        
        if result is None:
            print(f"      Scene {scene} is INVALID - excluded from valid_seq.json")
            continue
            
        if len(result) != 2:
            print(f"      Unexpected result format, skipping scene")
            continue
            
        valid_frames_with_annotations, frame_mapping = result
        
        # Keep only the frames that have annotations
        frames_to_process = [(num, name) for num, name in existing_frames 
                           if num in valid_frames_with_annotations]
        
        # Ensure normalized filenames
        valid_frames = ensure_normalized_filenames(scene_path, frames_to_process, dry_run)
        
        # Generate .npz files
        generate_npz_files(scene_path, scene, valid_frames_with_annotations, frame_data_by_seq, frame_mapping, dry_run)
        
        scenes_data[scene] = valid_frames_with_annotations
    
    # Generate valid_seq.json
    generate_valid_seq_json(category_path, scenes_data, dry_run)
    
    # Generate analysis report
    generate_analysis_report(category_path, category, coverage_stats, dry_run)
    
    print(f"  Completed category {category} with {len(scenes_data)} valid scenes")


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if not osp.exists(args.co3d_root):
        print(f"Error: CO3D root directory does not exist: {args.co3d_root}")
        return
    
    # Find categories to process
    if args.categories:
        categories = args.categories
    else:
        # Use predefined categories and filter by what exists
        available_categories = find_categories(args.co3d_root)
        categories = [cat for cat in CATEGORIES if cat in available_categories]
        
        if len(categories) != len(CATEGORIES):
            missing = set(CATEGORIES) - set(categories)
            print(f"Warning: Some predefined categories not found: {missing}")
    
    if not categories:
        print("No valid categories found!")
        return
    
    print(f"Processing categories: {categories}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE - No files will be modified ===")
    
    # Process each category
    for category in categories:
        category_path = osp.join(args.co3d_root, category)
        if not osp.exists(category_path):
            print(f"Warning: Category directory does not exist: {category_path}")
            continue
        
        process_category(args.co3d_root, category, args.dry_run)
    
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
    # python3 -m scripts.python_files.co3d_dataset_preprocess --co3d_root datasets/CO3D --dry_run
import os
from os import makedirs
from os.path import join
import torch
from torch.utils.data import DataLoader
import argparse
from datasets import InferenceDataset
from models import get_resnet
from tqdm import tqdm
import pandas as pd


def evaluate(args):
    print(f"infer images at {args.infer_path}")
    # Use a simple method name based on the folder name
    makedirs(join(args.output_dir), exist_ok=True)

    test_set = InferenceDataset(args)
    test_loader = DataLoader(test_set, num_workers=8, shuffle=False, batch_size=args.batch_size)
    model = get_resnet(args)
    model.load_state_dict(torch.load(args.ckpt_pth))

    model = model.to(device)
    model.eval()

    file_names = []
    art_scores = []

    with torch.no_grad():
        for item in tqdm(test_loader):
            cur_batch_files = item['image_file_name']
            cur_batch_scores = model(item['image'].to(device))
            file_names += cur_batch_files
            art_scores.append(cur_batch_scores.detach().cpu())

    art_scores = torch.cat(art_scores).numpy()
    art_scores = [float(i) for i in art_scores]

    result = pd.DataFrame(
        {
            'file_name': file_names,
            'art_score': art_scores,
        }
    )
    result.to_csv(os.path.join(args.output_dir, args.output_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArtScore Inference Script')
    # model config
    parser.add_argument('--auto_parse', action='store_true', help='auto parse model config from checkpoint name')
    parser.add_argument(
        '--ckpt_pth',
        type=str,
        required=True,
        help='Direct path to the checkpoint file'
    )
    parser.add_argument(
        '--no_dense_layer',
        help='if provided, no dense layer in fc',
        action='store_true'
    )
    parser.add_argument('--backbone_config', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    # inference config
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='batch size'
    )
    parser.add_argument(
        '--infer_path',
        type=str,
        required=True,
        help='Direct path to the folder containing evaluation images'
    )
    parser.add_argument(
        '--output_dir',
        default='./eval_results',
        type=str,
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='results.csv',
        help='Name of the output CSV file'
    )

    args = parser.parse_args()
    if args.auto_parse:
        args.backbone_config = 'resnet50' if 'resnet50' in args.ckpt_pth else 'resnet101'
        args.no_dense_layer = False if 'denseLayer@True' in args.ckpt_pth else True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate(args)

from os import makedirs
from os.path import join
import torch
import argparse
from datasets import ReferenceDataset
from torch.utils.data import DataLoader
from models import LPIPS, L2, SSIM_, GramLoss, ContentLoss
from tqdm import tqdm
import pandas as pd

metric_model_mapping = {
    'lpips': LPIPS,
    'l2': L2,
    'ssim': SSIM_,
    'gram': GramLoss,
    'content': ContentLoss
}

index_method_mapping = {
    0: '_0_AdaAttN',
    1: '_1_Arbitrary-Style-Transfer-via-Multi-Adaptation-Network',
    2: '_2_Avatar-Net_Pytorch',
    3: '_3_conditional-style-transfer',
    4: '_4_IEContraAST',
    5: '_5_LinearStyleTransfer',
    6: '_6_MAST',
    7: '_7_PAMA',
    8: '_8_Pytorch_WCT',
    9: '_9_pytorch-AdaIN',
    10: '_10_SANET',
    11: '_11_pytorch-neural-style-transfer',
    # 12: '_12_pytorch-neural-style-transfer',
}


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    method_name = index_method_mapping[args.stylized_method_index]
    print(f'used method: {method_name}')
    print(f'eval metric: {args.metric}')
    args.transfered_dir = join(args.stylized_method_root, method_name)
    test_set = ReferenceDataset(args)
    test_loader = DataLoader(test_set, num_workers=8, shuffle=False, batch_size=args.batch_size)
    model = metric_model_mapping[args.metric]()
    try:
        model = model.to(device)
        model.eval()
    except:
        pass
    file_names = []
    metric_scores = []
    with torch.no_grad():
        for item in tqdm(test_loader, total=len(test_loader)):
            cur_batch_files = item['file_name']
            x = item['trans'].to(device)
            y = item['ref'].to(device)
            cur_batch_scores = model(x, y)
            file_names += cur_batch_files
            metric_scores.append(cur_batch_scores.detach().cpu())
    metric_scores = torch.cat(metric_scores).numpy()
    metric_scores = [float(i) for i in metric_scores]
    result = pd.DataFrame(
        {
            'file_name': file_names,
            f'{args.metric}': metric_scores,
        }
    )
    makedirs(join(args.output_dir, args.metric), exist_ok=True)
    result.to_csv(join(args.output_dir, args.metric, f"{'_'.join(method_name.split('_')[2:])}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training procedure args')
    parser.add_argument(
        '--metric',
        help='which metric to measure',
        choices=['lpips', 'l2', 'ssim', 'gram', 'content'],
        type=str
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='batch size')
    parser.add_argument(
        '--output_dir',
        default='./all_metrics/',
        type=str,
    )
    parser.add_argument(
        '--style_dir',
        default='./art-fid/style',
        type=str,
    )
    parser.add_argument(
        '--content_dir',
        default='./art-fid/content',
        type=str,
    )
    parser.add_argument(
        '--stylized_method_root',
        type=str,
        default='./art-fid/outs'
    )
    parser.add_argument(
        '--stylized_method_index',
        type=int,
        default=0
    )

    args = parser.parse_args()
    evaluate(args)

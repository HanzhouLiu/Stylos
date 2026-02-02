from models import get_resnet, get_loss_function
from datasets import get_dataset, TrainDataset, TrainDatasetShuffled
from PIL import Image, ImageFile
from utils import set_seed, count_parameters, compute_metrics
from torch.optim import AdamW
import torch
import wandb
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
torch.multiprocessing.set_sharing_strategy('file_system')
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else 'cpu'

def val(args, model, loader):
    model.eval()
    epoch_logits = []
    epoch_labels = []
    paintings = []
    with torch.no_grad():
        print('evaluating...')
        for idx, item in enumerate(tqdm(loader)):
            cur_batch_size = item['rank'].size()[0]
            labels = item['rank']
            # flat then stack
            flat_item = item['image'].view(-1, 3, 224, 224).to(device)
            preds = model(flat_item)
            preds = preds.view(cur_batch_size, -1)
            epoch_logits.append(preds.cpu().data)
            epoch_labels.append(labels.cpu().data)
            paintings.append(item['painting'].cpu().data)
    ground_truth = torch.cat(epoch_labels)
    predictions = torch.cat(epoch_logits)
    paintings = torch.cat(paintings)

    support = predictions.size()[0]
    eval_result = compute_metrics(predictions, ground_truth, paintings)
    loss_f = get_loss_function(args.loss)
    if args.loss in ['point', 'listMLE']:
        loss = loss_f(predictions, ground_truth.float())
    elif args.loss in ['pairLogist']:
        place_holder = torch.tensor([int(predictions.size()[1]) for _ in range(predictions.size()[0])])
        loss = loss_f(predictions, ground_truth, place_holder).mean()
    elif args.loss in ['pairLambda']:
        place_holder = torch.tensor([int(predictions.size()[1]) for _ in range(predictions.size()[0])])
        loss = loss_f(predictions, ground_truth.float(), place_holder).mean()
    else:
        raise Exception('check the name of your loss function')
    eval_result['loss'] = loss

    logging.info(f"""
    support:             {support}
    loss:                {loss}
    ndcg_score:          {eval_result['ndcg_score']}
    apr_score:           {eval_result['apr_score']}
    painting_score:      {eval_result['painting_score']}
    per_rank_rank:       {eval_result['per_rank_rank']}
    per_rank_score:      {eval_result['per_rank_score']}
    """)
    return eval_result

def train(args, model, train_loader, val_loader, test_loader):
    exp_name = f'loss@{args.loss}_model@{args.backbone_config}_denseLayer@{not args.no_dense_layer}' \
               f'_batch_size@{args.batch_size}_lr@{args.lr}_dropout@{args.dropout}'
    experiment = wandb.init(project=args.project_name, resume='allow', anonymous='must', name=exp_name)
    experiment.config.update(
        dict(
            loss=args.loss,
            model=args.backbone_config,
            denseLayer=not args.no_dense_layer,
            batchSize=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            shuffled=args.shuffled_series
            )
    )
    logging.info(f'''Starting training:
            loss:             {args.loss},
            model:            {args.backbone_config},
            denseLayer:       {not args.no_dense_layer},
            batchSize:        {args.batch_size},
            lr:               {args.lr},
            dropout:          {args.dropout},
            shuffled:         {args.shuffled_series},
            exp name:         {exp_name},
            #param:           {count_parameters(model)},
    ''')
    print('fc layer:')
    print(model.fc)

    best_loss = float('inf')
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_f = get_loss_function(args.loss)

    for epoch in range(1, int(1 + args.epoch)):
        model.train()
        epoch_loss = 0.0
        epoch_logits = []
        epoch_labels = []
        paintings = []

        print(f'----------------------starting epoch {epoch}----------------------')
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epoch}', unit='series') as pbar:
            for idx, item in enumerate(train_loader):
                cur_batch_size = item['rank'].size()[0]
                labels = item['rank'].to(device)
                # flat then stack
                flat_item = item['image'].view(-1, 3, 224, 224).to(device)
                preds = model(flat_item)
                preds = preds.view(cur_batch_size, -1)
                # different loss function takes different input
                if args.loss in ['point', 'listMLE']:
                    loss = loss_f(preds, labels.float())
                elif args.loss in ['pairLogist']:
                    place_holder = torch.tensor([int(preds.size()[1]) for _ in range(preds.size()[0])]).to(device)
                    loss = loss_f(preds, labels, place_holder).mean()
                elif args.loss in ['pairLambda']:
                    place_holder = torch.tensor([int(preds.size()[1]) for _ in range(preds.size()[0])]).to(device)
                    loss = loss_f(preds, labels.float(), place_holder).mean()
                else:
                    raise Exception('check the name of your loss function')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().data
                epoch_logits.append(preds.cpu().data)
                epoch_labels.append(labels.cpu().data)
                paintings.append(item['painting'].cpu().data)
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                experiment.log({
                    'train loss': loss.item(),
                    'epoch': epoch
                })


        ground_truth = torch.cat(epoch_labels)
        predictions = torch.cat(epoch_logits)
        paintings = torch.cat(paintings)

        train_result = compute_metrics(predictions, ground_truth, paintings)
        train_result['loss'] = epoch_loss
        logging.info(f"""-----------
            performance on train set:
            loss:               {epoch_loss}
            ndcg_score:         {train_result['ndcg_score']}
            apr_score:          {train_result['apr_score']}
            painting_score:     {train_result['painting_score']}
            per_rank_rank:      {train_result['per_rank_rank']}
            per_rank_score:     {train_result['per_rank_score']}
            """)

        logging.info(f"""-----------
            performance on val set:""")
        val_result = val(args, model, val_loader)

        logging.info(f"""-----------
            performance on test set:""")
        test_result = val(args, model, test_loader)
        try:
            experiment.log({
                'Loss/train': train_result['loss'],
                'NDCG_score/train': train_result['ndcg_score'],
                'APR_score/train': train_result['apr_score'],
                'Painting_score/train': train_result['painting_score'],
                'Per_rank_rank/train': {
                    f'rank_{i}': r for i, r in enumerate(train_result['per_rank_rank'])
                },
                'Per_rank_score/train': {
                    f'rank_{i}': s for i, s in enumerate(train_result['per_rank_score'])
                },
                'Loss/val': val_result['loss'],
                'NDCG_score/val': val_result['ndcg_score'],
                'APR_score/val': val_result['apr_score'],
                'Painting_score/val': val_result['painting_score'],
                'Per_rank_rank/val': {
                    f'rank_{i}': r for i, r in enumerate(val_result['per_rank_rank'])
                },
                'Per_rank_score/val': {
                    f'rank_{i}': s for i, s in enumerate(val_result['per_rank_score'])
                },
                'Loss/test': test_result['loss'],
                'NDCG_score/test': test_result['ndcg_score'],
                'APR_score/test': test_result['apr_score'],
                'Painting_score/test': test_result['painting_score'],
                'Per_rank_rank/test': {
                    f'rank_{i}': r for i, r in enumerate(test_result['per_rank_rank'])
                },
                'Per_rank_score/test': {
                    f'rank_{i}': s for i, s in enumerate(test_result['per_rank_score'])
                },
            })
        except:
            pass

        if args.save_ckpt and val_result['loss'] < best_loss:
            print('saving model...')
            best_loss = val_result['loss']
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{exp_name}_best.pth'))
    print(f'training finished')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training procedure args')
    parser.add_argument(
        '--label_path',
        help='path to label file',
        default='face_dataset.csv',
        type=str,
    )
    parser.add_argument(
        '--backbone_config',
        help='eg., resnet101',
        default='resnet101',
        choices=['resnet101', 'resnet50'],
        type=str,
    )
    parser.add_argument(
        '--no_dense_layer',
        help='if provided, no dense layer in fc',
        action='store_true'
    )
    parser.add_argument("--device", type=str)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for reproduction')
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='running n epochs')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=1e-4,
        type=float)
    parser.add_argument(
        '--dropout',
        help='dropout rate',
        default=0.5,
        type=float)
    parser.add_argument(
        '--dont_save_model',
        action='store_false',
        dest='save_ckpt',
        help='if provided, do not save the model after training'
    )
    parser.add_argument(
        '--loss',
        choices=['point', 'pairLogist', 'pairLambda', 'listMLE'],
        type=str,
        required=True,
        help='the loss function to optimize'
    )
    parser.add_argument(
        '--output_dir',
        default='ckpt',
        type=str,
    )
    parser.add_argument(
        '--shuffled_series',
        action='store_true',
        help='if provided, train the model with shuffled series of images',
    )
    parser.add_argument(
        '--project_name',
        default='ArtScore',
        type=str,
    )
    args = parser.parse_args()
    set_seed(args.seed)
    args.device = device

    os.makedirs(args.output_dir, exist_ok=True)

    print('getting dataset...')
    if args.shuffled_series:
        print(f'shuffling series...')
        train_dataset, val_dataset, test_dataset = get_dataset(args, TrainDatasetShuffled)
    else:
        train_dataset, val_dataset, test_dataset = get_dataset(args, TrainDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # init model
    model = get_resnet(args)
    model = model.to(device)
    print(f'using device: {device}')
    train(args, model, train_loader, val_loader, test_loader)

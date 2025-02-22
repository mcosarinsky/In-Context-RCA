import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import shutil
import json
import medpy.metric.binary as metrics
import torchvision

from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm
from pathlib import Path
from src.utils.data_transforms import Scale, HUScale, ToTensor, OneHot
from src.utils.io import save_segmentation
from src.models.unet import UNet
from src.datasets import TrainerDataset


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs and targets shape is (N,C,H,W)
        dims = (1, 2, 3)
        intersection = torch.sum(inputs * targets, dims)
        cardinality = torch.sum(inputs + targets, dims)
        dice_score = (2. * intersection + smooth)/(cardinality + smooth)

        return torch.mean(1. - dice_score)


def compute_dice_multi(pred, target):
    num_classes = target.shape[0] # target is one-hot encoding of shape (C,H,W)
    dice_scores = []

    for i in range(1, num_classes):
        dice_scores.append(metrics.dc(pred==i, target[i]))

    return dice_scores


def trainer(train_dataset, val_dataset, model, args):
    torch.manual_seed(42)
    num_classes = val_dataset[0]['GT'].size(0)
    num_classes_no_background = num_classes - 1 if num_classes > 1 else num_classes
    print(f'Number of classes: {num_classes_no_background}\n')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dice_loss = DiceLoss().to(device)

    train_kwargs = {'pin_memory': True, 'batch_size': args['batch_size'], 'shuffle': True}
    val_kwargs = {'pin_memory': True, 'batch_size': 1, 'shuffle': False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], weight_decay=args['weight_decay'])

    output_dir = args['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    scores = []
    for epoch in tqdm(range(args['epochs']), initial=1):
        model.train()
        train_loss_avg = 0
        num_batches = 0

        for batch_idx, sample_batched in enumerate(train_loader):
            print(f' Batch {batch_idx+1}/{len(train_loader)}', end='\r')
            image, label = sample_batched['image'].to(device), sample_batched['GT'].to(device)
            outseg = model(image)
            optimizer.zero_grad()
            loss = dice_loss(outseg, label)
            train_loss_avg += loss.item()
            loss.backward()
            optimizer.step()
            num_batches += 1
            torch.cuda.empty_cache()

        train_loss_avg /= num_batches

        # We save segmentations every n_save iterations
        if (epoch+1) % args['n_save'] == 0:
            print(f'\nTrain average loss (1 - Dice): {train_loss_avg:.3f}\n')

            model.eval()
            num_batches = 0
            val_dice_avg = np.zeros(num_classes-1) if num_classes > 1 else 0

            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(val_loader):
                    image, label = sample_batched['image'].to(device), sample_batched['GT'].to(device)
                    seg = model(image)

                    if num_classes == 1:
                        out = seg[0,0,:,:] > 0.5 
                        val_dice_avg += metrics.dc(out.cpu().numpy(), label.cpu().numpy())
                    else:
                        out = torch.argmax(seg[0,:,:,:], axis = 0) 
                        classes_dice = compute_dice_multi(out.cpu().numpy(), label.squeeze(0).cpu().numpy()) 
                        val_dice_avg += classes_dice

                    num_batches += 1

                    if torch.unique(out).numel() != 1:
                        img_name = Path(sample_batched['name'][0]).stem
                        file_name = f'{img_name}_epoch{epoch+1}.png'
                        save_segmentation(out.cpu(), file_name, output_dir)
                        if num_classes == 1:
                            dice_score = metrics.dc(out.cpu().numpy(), label.cpu().numpy())
                        else:
                            dice_score = compute_dice_multi(out.cpu().numpy(), label.squeeze(0).cpu().numpy())
                        scores.append(dice_score)

                    torch.cuda.empty_cache()

            val_dice_avg /= num_batches
            print(f'\nValidation average dice coefficient: {np.round(val_dice_avg, 3)}\n')

    return scores


def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="Train UNet with given arguments")
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8, required=False)
    parser.add_argument('--lr', type=float, default=2.5e-4, required=False)
    parser.add_argument('--epochs', type=int, default=10, required=False)
    args = parser.parse_args()
    
    if args.dataset == 'camus':
        n_classes = 3
    elif args.dataset in ['psfhs', 'jsrt']:
        n_classes = 2
    else:
        n_classes = 1

    grayscale = True
    target_size = 128 if '3d-ircadb' in args.dataset else 256

    transforms_list = []
    transforms_list.extend([ToTensor(), OneHot(n_classes=n_classes)])
    transforms = torchvision.transforms.Compose(transforms_list)

    train_dataset = TrainerDataset(split='Train', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size, transform_img=transform_img)
    test_dataset = TrainerDataset(split='Test', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size, transform_img=transform_img)

    # Define training arguments
    trainer_args = {'batch_size': args.batch_size, 
                    'lr': args.lr, 
                    'weight_decay': 1e-5, 
                    'epochs': args.epochs, 
                    'n_save': 1, 
                    'output_dir': f'datasets/UNet/{args.out}'
                    }

    n_channels = 1 if grayscale else 3
    model = UNet(n_channels=n_channels, n_classes=1, batch_norm=True).float()
    print(f'Training with batch size {trainer_args['batch_size']} and learning rate {trainer_args['lr']}')

    # Train the model and save scores
    scores = trainer(train_dataset, test_dataset, model, trainer_args)
    
    with open(f"{trainer_args['output_dir']}/scores.json", 'w') as f:
        json.dump(scores, f)

if __name__ == "__main__":
    main()
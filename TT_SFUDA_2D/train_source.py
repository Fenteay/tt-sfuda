import os
import argparse
import yaml
from glob import glob
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import Dataset
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize

import archs
import losses
from metrics import iou_score
from utils import AverageMeter
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hrf', help='dataset name (e.g., hrf, rite, chase)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    args = parser.parse_args()
    return args

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou, dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return avg_meters

def main():
    args = parse_args()
    cudnn.benchmark = True

    # Common Configuration
    config = {
        'arch': 'UNet',
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'name': f"{args.dataset}_unet",
        'img_ext': '.png',
        'mask_ext': '.png',
        'input_h': 512,
        'input_w': 512,
        'num_workers': 4,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'loss': 'BCEDiceLoss',
        'stage1': 15,
        'stage2': 15
    }

    print(f"Loading {args.dataset} dataset...")
    train_img_ids = glob(os.path.join('inputs', 'inputs', args.dataset, 'train', 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', 'inputs', args.dataset, 'train', 'images'),
        mask_dir=os.path.join('inputs', 'inputs', args.dataset, 'train', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    print("Creating model %s..." % config['arch'])
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision']
    )
    model = model.to(device)

    criterion = losses.__dict__[config['loss']]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    print(f"Training Source Model on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        train(train_loader, model, criterion, optimizer)

    # Save logic
    model_dir = os.path.join('models', config['name'])
    os.makedirs(model_dir, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    # Save a generic config file so the target script works 
    # (Since tt_sfuda reads config_rite.yml, config_hrf.yml, etc based on the target)
    for target_dataset in ['rite', 'hrf', 'chase']:
        config_path = os.path.join(model_dir, f'config_{target_dataset}.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
    print(f"Finished training. Model and configurations saved to {model_dir}/")
    print(f"You can now run SFUDA adaptation using: python tt_sfuda_2d.py --source {config['name']} --target <target_dataset>")

if __name__ == '__main__':
    main()

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
    parser.add_argument('--dataset', default='leaf',
                        help='dataset name (e.g., leaf, hrf, rite, chase)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--input_size', default=256, type=int,
                        help='Input image size (default 256 for leaf, 512 for retinal)')
    parser.add_argument('--target', default='greenhouse',
                        help='Target domain dataset name for saving config')
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

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            iou, dice = iou_score(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            postfix = OrderedDict([('val_loss', avg_meters['loss'].avg), ('val_iou', avg_meters['iou'].avg)])
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
        'input_h': args.input_size,
        'input_w': args.input_size,
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

    val_img_ids = glob(os.path.join('inputs', 'inputs', args.dataset, 'val', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    print(f"  Train: {len(train_img_ids)} images | Val: {len(val_img_ids)} images")

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
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

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'inputs', args.dataset, 'val', 'images'),
        mask_dir=os.path.join('inputs', 'inputs', args.dataset, 'val', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
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
    best_iou = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        train_log = train(train_loader, model, criterion, optimizer)

        # Validate nếu có val set
        if len(val_img_ids) > 0:
            val_log = validate(val_loader, model, criterion)
            print(f"  train_loss: {train_log['loss'].avg:.4f} | train_iou: {train_log['iou'].avg:.4f} "
                  f"| val_loss: {val_log['loss'].avg:.4f} | val_iou: {val_log['iou'].avg:.4f}")
            current_iou = val_log['iou'].avg
        else:
            current_iou = train_log['iou'].avg

        # Lưu best model
        if current_iou > best_iou:
            best_iou = current_iou
            model_dir = os.path.join('models', config['name'])
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
            print(f"  ✅ Best model saved! IoU: {best_iou:.4f}")

    # Lưu config cho target domain(s)
    model_dir = os.path.join('models', config['name'])
    os.makedirs(model_dir, exist_ok=True)

    # Lưu model cuối nếu chưa có
    final_model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(final_model_path):
        torch.save(model.state_dict(), final_model_path)

    # Các target dataset có thể adapt sang
    target_datasets = [args.target, 'rite', 'hrf', 'chase']
    for target_dataset in set(target_datasets):
        config_path = os.path.join(model_dir, f'config_{target_dataset}.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"\nFinished training! Best IoU: {best_iou:.4f}")
    print(f"Model and configurations saved to {model_dir}/")
    print(f"\nRun SFUDA adaptation:")
    print(f"  python tt_sfuda_2d.py --source {config['name']} --target {args.target}")

if __name__ == '__main__':
    main()

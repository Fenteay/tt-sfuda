import os
import yaml
import torch
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from torch.utils.data import DataLoader

import archs
# IMPORT TRỰC TIẾP LỚP DATASET TỪ CODE CỦA TÁC GIẢ
from dataset import Dataset 

def main():
    # --- CẤU HÌNH ---
    TARGET = 'rite'
    SOURCE = 'hrf_unet'
    WEIGHTS_PATH = 'adapted_target_model_rite.pth'
    CONFIG_PATH = f'models/{SOURCE}/config_{TARGET}.yml'
    
    OUTPUT_MASK_DIR = 'batch_predictions'
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    device = torch.device('cpu') # Tiếp tục chạy CPU cho an toàn

    # 1. Load Cấu hình
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 2. Khởi tạo Mô hình
    print("1. Đang load mô hình...")
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Sử dụng chính xác cấu trúc DataLoader của tác giả
    print("2. Đang chuẩn bị DataLoader chuẩn của tác giả...")
    val_img_ids = glob(os.path.join('inputs', 'inputs', TARGET, 'test', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'inputs', TARGET, 'test', 'images'),
        mask_dir=os.path.join('inputs', 'inputs', TARGET, 'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0, # Set 0 khi chạy CPU để tránh lỗi đa luồng
        drop_last=False)

    # 4. Bắt đầu dự đoán
    print(f"3. Đang phân vùng {len(val_loader)} ảnh...\n")
    with torch.no_grad():
        for i, (input_tensor, target, meta) in enumerate(tqdm(val_loader, desc="Tiến trình")):
            input_tensor = input_tensor.to(device)
            
            # Dự đoán
            output = model(input_tensor)
            pred_prob = torch.sigmoid(output)
            pred_mask = (pred_prob > 0.5).float()
            
            # Chuyển thành ảnh
            mask_np = pred_mask[0].squeeze().cpu().numpy()
            final_mask = (mask_np * 255).astype(np.uint8)
            
            # Lưu ảnh
            img_name = val_img_ids[i] + ".png"
            save_path = os.path.join(OUTPUT_MASK_DIR, img_name)
            cv2.imwrite(save_path, final_mask)
            
    print(f"\nHoàn tất! Hãy kiểm tra thư mục: {OUTPUT_MASK_DIR}")

if __name__ == '__main__':
    main()
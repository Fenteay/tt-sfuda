# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
"""
prepare_leaf_dataset.py
-----------------------
Chuyển đổi Leaf Disease Segmentation Dataset (Kaggle) sang format TT-SFUDA_2D.

Cấu trúc nguồn (leaf dataset):
    dataset/leaf/data/data/images/  ← ảnh gốc (.jpg)
    dataset/leaf/data/data/masks/   ← mask binary (.png)

Cấu trúc đích (TT-SFUDA format):
    TT_SFUDA_2D/inputs/inputs/leaf/train/images/  (.png)
    TT_SFUDA_2D/inputs/inputs/leaf/train/masks/0/ (.png)
    TT_SFUDA_2D/inputs/inputs/leaf/val/images/
    TT_SFUDA_2D/inputs/inputs/leaf/val/masks/0/
    TT_SFUDA_2D/inputs/inputs/leaf/test/images/
    TT_SFUDA_2D/inputs/inputs/leaf/test/masks/0/

Cấu trúc đích cho greenhouse (target - không cần mask):
    TT_SFUDA_2D/inputs/inputs/greenhouse/train/images/
    TT_SFUDA_2D/inputs/inputs/greenhouse/train/masks/0/  ← dummy masks (zeros)
    TT_SFUDA_2D/inputs/inputs/greenhouse/test/images/
    TT_SFUDA_2D/inputs/inputs/greenhouse/test/masks/0/   ← dummy masks (zeros)

Cách dùng:
    python prepare_leaf_dataset.py
"""

import os
import shutil
import random
import cv2
import numpy as np
from glob import glob

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN - chỉnh sửa nếu cần
# ============================================================
# Thư mục gốc workspace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset nguồn (leaf)
LEAF_IMG_DIR  = r"c:\Users\lhuynh35\OneDrive - DXC Production\Documents\VsCode\Python\dataset\leaf\data\data\images"
LEAF_MASK_DIR = r"c:\Users\lhuynh35\OneDrive - DXC Production\Documents\VsCode\Python\dataset\leaf\data\data\masks"

# Dataset greenhouse (target)
GREENHOUSE_DIR = r"c:\Users\lhuynh35\OneDrive - DXC Production\Documents\VsCode\Python\dataset\greenhouse\subset"

# Thư mục đích trong TT-SFUDA
OUTPUT_DIR = os.path.join(BASE_DIR, "inputs", "inputs")

# Tỉ lệ split: train / val / test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Seed cho reproducibility
RANDOM_SEED = 42

# ============================================================

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def copy_as_png(src_path, dst_path):
    """Đọc ảnh và lưu dạng PNG (convert nếu cần)."""
    img = cv2.imread(src_path)
    if img is None:
        print(f"  [WARN] Không đọc được: {src_path}")
        return False
    cv2.imwrite(dst_path, img)
    return True

def binarize_mask(mask_path, dst_path):
    """
    Đọc mask, binarize (0/255) và lưu PNG grayscale.
    TT-SFUDA đọc mask chia 255 → giá trị 0.0 hoặc 1.0.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [WARN] Không đọc được mask: {mask_path}")
        return False
    # Binarize: pixel > 127 → 255, còn lại → 0
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(dst_path, mask_bin)
    return True

def create_dummy_mask(ref_img_path, dst_path, value=0):
    """Tạo mask rỗng (zeros) cho target domain (greenhouse)."""
    img = cv2.imread(ref_img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    dummy = np.full((h, w), value, dtype=np.uint8)
    cv2.imwrite(dst_path, dummy)
    return True

# ============================================================
# BƯỚC 1: Chuẩn bị LEAF dataset (source domain)
# ============================================================
def prepare_leaf(leaf_img_dir, leaf_mask_dir, output_dir):
    print("\n" + "="*60)
    print("BƯỚC 1: Chuẩn bị LEAF dataset (Source Domain)")
    print("="*60)

    # Lấy danh sách tất cả ảnh
    all_imgs = sorted(glob(os.path.join(leaf_img_dir, "*.jpg")))
    all_ids  = [os.path.splitext(os.path.basename(p))[0] for p in all_imgs]

    # Kiểm tra mask tương ứng
    valid_ids = []
    for img_id in all_ids:
        mask_path = os.path.join(leaf_mask_dir, img_id + ".png")
        if os.path.exists(mask_path):
            valid_ids.append(img_id)
        else:
            print(f"  [WARN] Không có mask cho: {img_id}")

    print(f"  Tổng ảnh hợp lệ: {len(valid_ids)}")

    # Shuffle và split
    random.seed(RANDOM_SEED)
    random.shuffle(valid_ids)

    n = len(valid_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_ids = valid_ids[:n_train]
    val_ids   = valid_ids[n_train:n_train + n_val]
    test_ids  = valid_ids[n_train + n_val:]

    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Copy files theo từng split
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        img_out  = os.path.join(output_dir, "leaf", split_name, "images")
        mask_out = os.path.join(output_dir, "leaf", split_name, "masks", "0")
        make_dirs(img_out)
        make_dirs(mask_out)

        ok_count = 0
        for img_id in split_ids:
            src_img  = os.path.join(leaf_img_dir,  img_id + ".jpg")
            src_mask = os.path.join(leaf_mask_dir, img_id + ".png")
            dst_img  = os.path.join(img_out,  img_id + ".png")
            dst_mask = os.path.join(mask_out, img_id + ".png")

            if copy_as_png(src_img, dst_img) and binarize_mask(src_mask, dst_mask):
                ok_count += 1

        print(f"  [{split_name.upper()}] Đã copy: {ok_count}/{len(split_ids)} ảnh")

    print("  ✅ Hoàn thành LEAF dataset!")

# ============================================================
# BƯỚC 2: Chuẩn bị GREENHOUSE dataset (target domain)
# ============================================================
def prepare_greenhouse(greenhouse_dir, output_dir):
    print("\n" + "="*60)
    print("BƯỚC 2: Chuẩn bị GREENHOUSE dataset (Target Domain)")
    print("="*60)

    # Greenhouse có 2 split folders: train, val, test
    # Mỗi split có thư mục theo class (class-level)
    # Chúng ta gom tất cả ảnh lại, không phân biệt class
    # vì TT-SFUDA là unsupervised (không cần label)

    all_img_paths = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(greenhouse_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            if os.path.isdir(class_path):
                imgs = glob(os.path.join(class_path, "*.jpg")) + \
                       glob(os.path.join(class_path, "*.png")) + \
                       glob(os.path.join(class_path, "*.jpeg"))
                all_img_paths.extend(imgs)

    print(f"  Tổng ảnh greenhouse: {len(all_img_paths)}")

    if len(all_img_paths) == 0:
        print("  [ERROR] Không tìm thấy ảnh greenhouse!")
        return

    # Shuffle và split: 80% train, 20% test (không cần val vì unsupervised)
    random.seed(RANDOM_SEED)
    random.shuffle(all_img_paths)

    n = len(all_img_paths)
    n_train = int(n * 0.8)
    train_paths = all_img_paths[:n_train]
    test_paths  = all_img_paths[n_train:]

    print(f"  Train: {len(train_paths)} | Test: {len(test_paths)}")

    for split_name, split_paths in [("train", train_paths), ("test", test_paths)]:
        img_out  = os.path.join(output_dir, "greenhouse", split_name, "images")
        mask_out = os.path.join(output_dir, "greenhouse", split_name, "masks", "0")
        make_dirs(img_out)
        make_dirs(mask_out)

        ok_count = 0
        for i, src_img in enumerate(split_paths):
            img_id   = f"gh_{i:05d}"
            dst_img  = os.path.join(img_out,  img_id + ".png")
            dst_mask = os.path.join(mask_out, img_id + ".png")

            if copy_as_png(src_img, dst_img):
                create_dummy_mask(dst_img, dst_mask, value=0)
                ok_count += 1

        print(f"  [{split_name.upper()}] Đã copy: {ok_count}/{len(split_paths)} ảnh")

    print("  ✅ Hoàn thành GREENHOUSE dataset!")

# ============================================================
# BƯỚC 3: Tạo file config YAML cho TT-SFUDA
# ============================================================
def create_config(output_dir):
    print("\n" + "="*60)
    print("BƯỚC 3: Tạo config YAML")
    print("="*60)

    import yaml

    config = {
        'arch': 'UNet',
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'name': 'leaf_unet',
        'img_ext': '.png',
        'mask_ext': '.png',
        'input_h': 256,
        'input_w': 256,
        'num_workers': 4,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'loss': 'BCEDiceLoss',
        'stage1': 15,
        'stage2': 15
    }

    # Tạo thư mục model
    model_dir = os.path.join(os.path.dirname(output_dir), "models", "leaf_unet")
    make_dirs(model_dir)

    # Tạo config cho target = greenhouse
    config_path = os.path.join(model_dir, "config_greenhouse.yml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Config saved: {config_path}")
    print("  ✅ Hoàn thành tạo config!")

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  TT-SFUDA Dataset Preparation: Leaf -> Greenhouse")
    print("=" * 60)

    # Kiểm tra đường dẫn nguồn
    if not os.path.isdir(LEAF_IMG_DIR):
        print(f"[ERROR] Không tìm thấy LEAF images: {LEAF_IMG_DIR}")
        return
    if not os.path.isdir(LEAF_MASK_DIR):
        print(f"[ERROR] Không tìm thấy LEAF masks: {LEAF_MASK_DIR}")
        return
    if not os.path.isdir(GREENHOUSE_DIR):
        print(f"[ERROR] Không tìm thấy GREENHOUSE: {GREENHOUSE_DIR}")
        return

    prepare_leaf(LEAF_IMG_DIR, LEAF_MASK_DIR, OUTPUT_DIR)
    prepare_greenhouse(GREENHOUSE_DIR, OUTPUT_DIR)
    create_config(OUTPUT_DIR)

    print("\n" + "="*60)
    print("✅ HOÀN THÀNH! Cấu trúc dữ liệu đã sẵn sàng.")
    print("="*60)
    print("\nBước tiếp theo:")
    print("  1. Train source model:")
    print("     cd TT_SFUDA_2D")
    print("     python train_source.py --dataset leaf --epochs 50")
    print("\n  2. Adapt sang greenhouse (SFUDA):")
    print("     python tt_sfuda_2d.py --source leaf_unet --target greenhouse")
    print("="*60)

if __name__ == '__main__':
    main()

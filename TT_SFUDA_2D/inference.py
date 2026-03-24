import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

# =========================================================================
# 1. COPY CÁC IMPORT TỪ FILE tt_sfuda_2d.py VÀO ĐÂY
# (Ví dụ: import các file cấu trúc mạng UNet, hàm load dataloader RITE...)
# =========================================================================
# from networks.unet import UNet  <-- Thay bằng import thực tế của bạn
# from dataset.dataset import get_target_dataloader <-- Thay bằng import thực tế
# =========================================================================

def main():
    # --- CẤU HÌNH ---
    # Thay đường dẫn này bằng đường dẫn tới file .pth bạn tìm được ở Bước 1
    WEIGHTS_PATH = "checkpoints/target_model_rite.pth" 
    OUTPUT_DIR = "results/rite_inference_masks"
    
    # Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("1. Đang khởi tạo mô hình...")
    # =========================================================================
    # 2. KHỞI TẠO MODEL VÀ LOAD WEIGHTS
    # (Copy cách khởi tạo mạng student từ file tt_sfuda_2d.py)
    # =========================================================================
    # student = UNet(in_channels=3, out_classes=1) <-- Thay bằng code thực tế
    student = None # Xóa dòng này đi sau khi bạn copy code khởi tạo model vào
    
    # Load trọng số đã train vào model
    student.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    student.to(device)
    student.eval() # Bắt buộc phải có để chuyển sang chế độ test
    print("-> Load weights thành công!")

    print("2. Đang chuẩn bị dữ liệu test...")
    # =========================================================================
    # 3. KHỞI TẠO DATALOADER CHO TẬP TEST (RITE)
    # (Copy cách lấy target_test_loader từ file tt_sfuda_2d.py)
    # =========================================================================
    # target_test_loader = get_target_dataloader(dataset='rite', split='test') <-- Thay bằng code thực tế
    target_test_loader = [] # Xóa dòng này đi sau khi copy code lấy dataloader vào

    print(f"3. Bắt đầu dự đoán và lưu ảnh vào thư mục: {OUTPUT_DIR}")
    # Tắt tính toán gradient để tiết kiệm RAM và tăng tốc
    with torch.no_grad():
        for i, batch in enumerate(tqdm(target_test_loader)):
            # Lấy ảnh gốc (bạn check lại key 'image' hay 'img' trong batch nhé)
            images = batch['image'].to(device) 
            
            # Mô hình dự đoán
            outputs = student(images)
            
            # Xử lý output thành mask nhị phân (0 hoặc 1)
            pred_prob = torch.sigmoid(outputs)
            pred_mask = (pred_prob > 0.5).float()
            
            # Duyệt qua từng ảnh trong batch (nếu batch_size > 1)
            for j in range(images.size(0)):
                # Lấy tên ảnh để lưu (check lại key 'name' hay 'id' trong dataloader)
                if 'name' in batch:
                    img_name = batch['name'][j]
                else:
                    img_name = f"pred_img_{i}_{j}"
                
                save_name = f"{img_name}.png"
                
                # Chuyển tensor sang Numpy và scale lên 255 (ảnh trắng đen)
                mask_np = pred_mask[j].squeeze().cpu().numpy()
                final_mask = (mask_np * 255).astype(np.uint8)
                
                # Lưu ảnh bằng OpenCV
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, final_mask)
                
    print("\nHoàn tất! Hãy mở thư mục kết quả để kiểm tra.")

if __name__ == '__main__':
    main()
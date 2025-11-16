# 文件名: evaluate_single_image.py
# 描述: 加载模型，对单张带标签的图像进行推理，并计算 Dice Score

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 导入模型架构和评估函数
from model import Unet #
from utils import dice_score #

# --- 1. 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 必须与 train.py 中的设置完全一致
IMG_HEIGHT = 256
IMG_WIDTH = 256

# 指向你训练好的模型权重文件
MODEL_PATH = "best_cbam_unet_pets.pth.tar" #

# --- 指定你要测试的图像ID ---
# (不要带 .jpg 或 .png 后缀)
DATA_DIR = "./data"
TEST_IMAGE_ID = "Abyssinian_10" # 这是一个示例，你可以换成 test.txt 中的任何一个ID


def load_model(model_path, device):
    """加载模型和权重 (与 predict.py 相同)"""
    print(f"正在从 {model_path} 加载模型...")
    model = Unet(in_channels=3, num_classes=2).to(device) #
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"]) #
        model.eval() # 切换到评估模式
        print("模型加载成功并已设为评估模式。")
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 {model_path}")
        exit()
    return model

def get_transforms():
    """获取与 train.py 一致的变换"""
    img_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) #
    
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
    ]) #
    
    return img_transform, mask_transform

def load_image_and_mask(data_dir, image_id, img_transform, mask_transform):
    """加载单张图像和其对应的掩码"""
    
    # 1. 加载图像 (类似 predict.py)
    img_path = os.path.join(data_dir, "images", f"{image_id}.jpg") #
    try:
        image_pil = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 未找到图像文件 {img_path}")
        exit()
        
    input_tensor = img_transform(image_pil)
    input_batch = input_tensor.unsqueeze(0) # (1, C, H, W)
    
    # 2. 加载掩码 (类似 pet_dataset.py)
    mask_path = os.path.join(data_dir, "annotations", "trimaps", f"{image_id}.png") #
    try:
        mask_pil = Image.open(mask_path).convert("L")
    except FileNotFoundError:
        print(f"错误: 未找到掩码文件 {mask_path}")
        exit()

    mask_resized = mask_transform(mask_pil)
    
    # --- 关键预处理：转换掩码标签 (逻辑来自 pet_dataset.py) ---
    mask_np = np.array(mask_resized)
    target_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
    target_mask[mask_np == 1] = 1  # 类别 1 (Pet)
    target_mask[mask_np == 2] = 0  # 类别 0 (Background)
    target_mask[mask_np == 3] = 0  # 类别 0 (Border)

    target_tensor = torch.from_numpy(target_mask).long()
    target_batch = target_tensor.unsqueeze(0) # (1, H, W)

    return input_batch, target_batch

def main():
    # 1. 加载模型
    model = load_model(MODEL_PATH, DEVICE)
    
    # 2. 获取变换
    img_transform, mask_transform = get_transforms()
    
    # 3. 加载图像和对应的真值掩码
    print(f"正在加载图像和掩码: {TEST_IMAGE_ID}")
    input_tensor, target_tensor = load_image_and_mask(
        DATA_DIR, TEST_IMAGE_ID, img_transform, mask_transform
    )
    
    # 4. 将数据移至设备
    input_tensor = input_tensor.to(DEVICE)
    target_tensor = target_tensor.to(DEVICE) # 形状为 (1, H, W)
    
    # 5. 执行推理
    print("正在执行推理...")
    with torch.no_grad():
        logits = model(input_tensor) # 形状为 (1, C, H, W)
        
    # 6. 计算 Dice Score (前景)
    score = dice_score(logits, target_tensor) 
    
    # 7. 打印结果
    print("-" * 30)
    print(f"评估完成:")
    print(f"图像: {TEST_IMAGE_ID}")
    print(f"Dice Score (前景): {score.item() * 100:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    main()
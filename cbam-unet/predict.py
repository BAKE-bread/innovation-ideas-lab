# 文件名: predict.py
# 描述: 加载训练好的CBAM U-Net模型，对单张图像进行推理

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as rcParams
import numpy as np
import os

# 导入我们的模型架构
from model import Unet #

# --- 1. 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 必须与 train.py 中的设置完全一致
IMG_HEIGHT = 256
IMG_WIDTH = 256

# 指向你训练好的模型权重文件
MODEL_PATH = "best_cbam_unet_pets.pth.tar" 

# 指定你要测试的图像路径
# (确保这张图在你的 data/images/ 目录下)
#TEST_IMAGE_PATH = "./data/images/Abyssinian_10.jpg"
TEST_IMAGE_PATH = "Snipaste_2025-11-14_02-10-53.jpg"


def load_model(model_path, device):
    """加载模型和权重"""
    print(f"正在从 {model_path} 加载模型...")
    # 初始化模型架构
    model = Unet(in_channels=3, num_classes=2).to(device)
    
    # 加载检查点
    try:
        # 将 weights_only 设置为 True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval() # 切换到评估模式 (非常重要!)
        print("模型加载成功并已设为评估模式。")
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 {model_path}")
        exit()
    except KeyError:
        print("错误: 检查点文件格式不正确，可能缺少 'state_dict' 键。")
        exit()
        
    return model

def preprocess_image(image_path):
    """加载图像并应用与训练时相同的变换"""
    
    # 1. 定义变换 (必须与 train.py 一致)
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), #
        transforms.ToTensor(), #
        transforms.Normalize( #
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # 2. 加载图像
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 未找到图像文件 {image_path}")
        exit()
        
    original_pil = image.copy() # 保存原始PIL图像用于可视化

    # 3. 应用变换
    input_tensor = transform(image)
    
    # 4. 添加 Batch 维度 (C, H, W) -> (B, C, H, W)
    #
    input_batch = input_tensor.unsqueeze(0) 
    
    return input_batch, original_pil

def postprocess_and_visualize(original_pil, logits, original_path):
    """对模型输出进行后处理并可视化结果"""
    
    # logits 形状为 (B=1, C=2, H, W)
    
    # 1. Softmax 获取概率 (dim=1 是通道维度)
    probabilities = F.softmax(logits, dim=1)
    
    # 2. Argmax 获取每个像素的类别 (0 或 1)
    # (类似 utils.py 中的 dice_score)
    # pred_mask 形状为 (B=1, H, W)
    pred_mask = torch.argmax(probabilities, dim=1)
    
    # 3. 移除 Batch 维度并转到 CPU/Numpy
    # (B=1, H, W) -> (H, W)
    mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    
    # 4. 可视化
    print("正在生成可视化结果...")
    
    # 将原始图像也resize到(256, 256)，以便对齐显示
    original_resized = original_pil.resize((IMG_WIDTH, IMG_HEIGHT))

    fig, ax = plt.subplots(1, 3, figsize=(18, 7))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 子图1: 原图
    ax[0].imshow(original_resized)
    ax[0].set_title(f"原始图像: {os.path.basename(original_path)}")
    ax[0].axis("off")
    
    # 子图2: 预测掩码 (类别 1 = 宠物, 0 = 背景)
    ax[1].imshow(mask_np, cmap='gray') # 'gray'  cmap 使 0=黑, 1=白
    ax[1].set_title("预测掩码 (类别 1)")
    ax[1].axis("off")
    
    # 子图3: 叠加显示
    ax[2].imshow(original_resized)
    # 'jet' cmap 比较鲜艳, alpha=0.5 设置半透明
    ax[2].imshow(mask_np, cmap='jet', alpha=0.5) 
    ax[2].set_title("掩码叠加")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. 加载模型
    model = load_model(MODEL_PATH, DEVICE)
    
    # 2. 加载和预处理图像
    input_tensor, original_pil = preprocess_image(TEST_IMAGE_PATH)
    
    # 3. 将输入数据移至设备
    input_tensor = input_tensor.to(DEVICE)
    
    # 4. 执行推理
    print("正在执行推理...")
    with torch.no_grad(): # 关闭梯度计算 (节省内存和计算)
        logits = model(input_tensor)
        
    # 5. 后处理和可视化
    postprocess_and_visualize(original_pil, logits, TEST_IMAGE_PATH)


if __name__ == "__main__":
    main()
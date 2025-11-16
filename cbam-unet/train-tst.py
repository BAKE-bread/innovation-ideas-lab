# 文件名: train-tst.py
# 描述: 训练和验证 CBAM U-Net

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

# 导入我们自己的模块
from model import Unet
from pet_dataset import PetDataset
from utils import check_accuracy, dice_score

# --- 1. 超参数和配置 ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # (原测试为4)
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMG_HEIGHT = 256 # (原测试为256)
IMG_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False # 是否加载预训练模型
DATA_DIR = "./data" # 确保这指向你的数据根目录

# 官方的训练/验证/测试分割文件
TRAIN_SPLIT_FILE = "trainval.txt"
VAL_SPLIT_FILE = "test.txt"


def main():
    print(f"--- 正在使用设备: {DEVICE} ---")
    
    # --- 2. 定义图像和掩码的变换 ---
    # 图像变换：Resize, ToTensor, Normalize
    img_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 掩码变换：Resize (必须使用最近邻插值)
    # 注意：我们不在transform中转Tensor，而是在Dataset的__getitem__中
    # 手动处理，以确保标签映射正确
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # --- 3. 创建 Datasets 和 DataLoaders ---
    train_dataset = PetDataset(
        data_dir=DATA_DIR,
        split_file=TRAIN_SPLIT_FILE,
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = PetDataset(
        data_dir=DATA_DIR,
        split_file=VAL_SPLIT_FILE, # 使用官方 test.txt 作为我们的验证集
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False
    )

    # --- 4. 初始化模型、损失函数和优化器 ---
    # 模型来自 test_attention3.py
    # 输入3通道, 输出2类别 (0: 背景/边框, 1: 宠物)
    model = Unet(in_channels=3, num_classes=2).to(DEVICE)
    
    # 损失函数: 交叉熵
    # 我们的模型输出 (B, 2, H, W)
    # 我们的标签输入 (B, H, W) (值为 0 或 1)
    # 这是 CrossEntropyLoss 的标准用法
    loss_fn = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (可选) 加载检查点
    if LOAD_MODEL:
        try:
            checkpoint = torch.load("cbam_unet_pets.pth.tar")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> 已加载模型检查点")
        except FileNotFoundError:
            print("=> 未找到模型检查点，从头开始训练")


    # --- 5. 训练循环 ---
    print("--- 开始训练 ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train() # 设置为训练模式
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE) # targets 形状为 (B, H, W)

            # 1. 前向传播
            predictions = model(data) # predictions 形状为 (B, 2, H, W)
            
            # 2. 计算损失
            loss = loss_fn(predictions, targets)
            
            # 3. 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 4. 更新权重
            optimizer.step()
            
            # 更新tqdm的描述
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - 平均训练损失: {running_loss/len(train_loader):.4f}")

        # --- 6. 验证 ---
        check_accuracy(val_loader, model, device=DEVICE)

    # --- 7. 保存模型 ---
    print("--- 训练完成，正在保存模型 ---")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "cbam_unet_pets.pth.tar")


if __name__ == "__main__":
    main()
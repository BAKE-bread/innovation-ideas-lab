# 文件名: pet_dataset.py
# 描述: Oxford-IIIT Pet 数据集加载器

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PetDataset(Dataset):
    def __init__(self, data_dir, split_file, img_transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "annotations", "trimaps")
        
        self.file_list = []
        with open(os.path.join(data_dir, "annotations", split_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    self.file_list.append(parts[0]) # parts[0] is the image ID

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.file_list[idx]
        
        # 构造文件路径
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")

        # 加载图像和掩码
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L") # 确保为单通道
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            print(f"Attempted img_path: {img_path}")
            print(f"Attempted mask_path: {mask_path}")
            # 返回一个空值或处理错误
            return None, None


        # 应用Transform
        if self.img_transform:
            image = self.img_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        # --- 关键预处理：转换掩码标签 ---
        # 此时 mask 是 (1, H, W) 的张量，值为 1, 2, 3
        # 我们需要 (H, W) 的张量，值为 0, 1
        
        # 将PIL Image或Tensor转为Numpy进行高效处理
        # 注意：torchvision.transforms.ToTensor() 会将 (H,W,C) -> (C,H,W)
        # 并且会将 0-255 缩放到 0-1。我们不希望这样。
        # 因此，更好的做法是在transform中只做Resize，然后在这里处理
        
        # 假设 mask_transform 只是 Resize 和 ToPILImage()
        # 我们在 __init__ 中定义更合适的 transform
        
        # 将掩码转为Numpy数组
        mask_np = np.array(mask)
        
        # Pet=1, Background=2, Border=3
        target_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        target_mask[mask_np == 1] = 1  # 类别 1 (Pet)
        target_mask[mask_np == 2] = 0  # 类别 0 (Background)
        target_mask[mask_np == 3] = 0  # 类别 0 (Border)

        # 转换为 (H, W) 的 LongTensor，CrossEntropyLoss 需要这个格式
        target_mask = torch.from_numpy(target_mask).long()

        return image, target_mask
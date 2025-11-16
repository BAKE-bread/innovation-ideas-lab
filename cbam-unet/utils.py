# 文件名: utils.py
# 描述: 包含评估指标（如Dice Score）

import torch
import torch.nn.functional as F

def dice_score(pred, target, smooth=1e-5):
    """
    计算Dice Score (前景)
    pred: (B, C, H, W) - 模型的原始输出 (logits)
    target: (B, H, W) - 真实标签 (0, 1)
    """
    
    # 1. 获取预测的类别 (B, H, W)
    # 我们关心的是 类别 1 (Pet)
    pred_prob = F.softmax(pred, dim=1) # (B, C, H, W)
    pred_mask = pred_prob.argmax(dim=1) # (B, H, W)

    # 2. 将 pred_mask 和 target 展平
    pred_flat = pred_mask.contiguous().view(pred_mask.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)

    # 3. 计算交集 (Intersection) 和 并集 (Union)
    # 我们只计算前景（类别1）的Dice
    intersection = (pred_flat == 1) & (target_flat == 1)
    intersection = intersection.sum(dim=1).float()

    union = (pred_flat == 1).sum(dim=1).float() + (target_flat == 1).sum(dim=1).float()

    # 4. 计算Dice
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # 返回批次的平均Dice
    return dice.mean()

def check_accuracy(loader, model, device="cuda"):
    """
    在验证集上检查 Dice Score
    """
    num_correct = 0
    num_pixels = 0
    dice = 0.0
    model.eval() # 设置为评估模式

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device) # y is (B, H, W)
            
            preds = model(x) # preds is (B, C, H, W)
            dice += dice_score(preds, y)

    avg_dice = dice / len(loader)
    print(f"Validation Set: Average Dice Score (Foreground): {avg_dice*100:.2f}%")
    
    model.train() # 切换回训练模式
    return avg_dice
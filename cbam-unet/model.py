# 文件名: model.py
# 描述: 包含 CBAM 和 U-Net 架构
# 来源: test_attention3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 修正：CBAM的论文中使用了一个共享的MLP，这里为了简单起见
        # 保持了单层卷积
        # 如果需要更标准的CBAM，这里应该是一个两层的MLP
        self.conv = nn.Conv2d(channel, channel, 1, bias=False)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.conv(self.avg_pool(x))
        maxout = self.conv(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv2d(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x  
        out = self.spatial_attention(out) * out  
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Unet, self).__init__()
        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)

        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 双注意力模块（插入下采样之间）
        self.cbam1 = CBAM(64)  
        self.cbam2 = CBAM(128) 
        self.cbam3 = CBAM(256) 
        self.cbam4 = CBAM(512) 

        # 最终分类
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # 编码器（下采样阶段插入双注意力）
        e1 = self.enc1(x)
        e1 = self.cbam1(e1)  
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e2 = self.cbam2(e2)
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e3 = self.cbam3(e3)
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e4 = self.cbam4(e4)
        e5 = self.enc5(F.max_pool2d(e4, 2))

        # 解码器（上采样+特征融合）
        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 最终分类
        output = self.final_conv(d1)
        return output
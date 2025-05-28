import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):

    def __init__(self, channels, use_relu=True):
        super().__init__()
        # 通道对齐卷积（保持通道数不变）
        self.align_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 激活函数开关（与Metal版本ReLU对应）
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, a, b):
        """
        参数:
        a: 基准特征 [B, C, H, W]
        b: 待融合特征 [B, C, H', W'] (H' < H, W' < W)

        返回:
        fused: 融合后特征 [B, C, H, W]
        """
        # 特征对齐（1x1卷积）
        aligned_a = self.align_conv(a)

        # 双线性插值上采样b到a的尺寸
        resized_b = F.interpolate(b,
                                  size=a.shape[-2:],
                                  mode='bilinear',
                                  align_corners=True)

        # 特征融合（逐元素相加）
        fused = aligned_a + resized_b

        # 可选激活函数
        if self.relu is not None:
            fused = self.relu(fused)

        return fused


# 添加通道注意力机制
class FeatureFusionWithAttention(FeatureFusionModule):

    def __init__(self, channels):
        super().__init__(channels)
        # 通道注意力模块
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(channels, channels // 16, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(channels // 16, channels, 1),
                                       nn.Sigmoid())

    def forward(self, a, b):
        aligned_a = self.align_conv(a)
        resized_b = F.interpolate(b,
                                  a.shape[-2:],
                                  mode='bilinear',
                                  align_corners=True)

        # 生成注意力权重
        att = self.attention(aligned_a + resized_b)

        # 加权融合
        return att * aligned_a + (1 - att) * resized_b


#双线性插值下采样模板更新器
class AlignTemplateUpdater(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 尺寸对齐模块（保持stride=1）
        self.align_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        # 注意力生成模块
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1)  # 输出3个注意力图
        )

    def forward(self, a, b, c):
        """
        a: 基准模板 [B,C,H,W]
        b/c: 大尺寸模板 [B,C,2H,2W]
        """
        # 下采样b/c到a的尺寸（保持stride=1）
        b_down = F.interpolate(b,
                               size=a.shape[2:],
                               mode='bilinear',
                               align_corners=False)
        c_down = F.interpolate(c,
                               size=a.shape[2:],
                               mode='bilinear',
                               align_corners=False)

        # 对齐特征空间
        b_down = self.align_conv(b_down)
        c_down = self.align_conv(c_down)

        # 通道拼接
        combined = torch.cat([a, b_down, c_down], dim=1)  # [B,3C,H,W]

        # 生成注意力权重
        attn = self.attn_conv(combined)  # [B,3,H,W]
        attn = F.softmax(attn, dim=1)

        # 加权融合
        return a * attn[:, 0:1] + b_down * attn[:, 1:2] + c_down * attn[:, 2:3]

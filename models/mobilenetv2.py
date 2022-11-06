"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, mode='t', size=32, cates=100):
        super().__init__()
        
        if mode == 't':
            repeatlist=[2,4,6,23,3]
        else:
            repeatlist = [2,3,4,2,2]
        
        if size == 224 or size == 64:
            self.pre = nn.Sequential(
                nn.Conv2d(3, 32, stride=2, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True)
            ) # 112 / 32
        else:
            self.pre = nn.Sequential(
                nn.Conv2d(3, 32, stride=1, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True)
            ) # 32
        

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        if size == 64 or size == 32:
            self.stage2 = self._make_stage(repeatlist[0], 16, 24, 2, 6) # 16
            self.stage3 = self._make_stage(repeatlist[1], 24, 32, 2, 6) # 8
            self.stage4 = self._make_stage(repeatlist[2], 32, 64, 2, 6) # 4
            self.stage5 = self._make_stage(repeatlist[3], 64, 96, 1, 6)
            self.stage6 = self._make_stage(repeatlist[4], 96, 160, 1, 6)
        else:
            self.stage2 = self._make_stage(repeatlist[0], 16, 24, 2, 6) # 56
            self.stage3 = self._make_stage(repeatlist[1], 24, 32, 2, 6) # 28
            self.stage4 = self._make_stage(repeatlist[2], 32, 64, 2, 6) # 14
            self.stage5 = self._make_stage(repeatlist[3], 64, 96, 2, 6) # 7
            self.stage6 = self._make_stage(repeatlist[4], 96, 160, 1, 6)
        
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, cates, 1)

    def forward(self, x):
        x = self.pre(x)
        x0 = self.stage1(x)
        x1 = self.stage2(x0)
        x2 = self.stage3(x1)
        x3 = self.stage4(x2)
        x4 = self.stage5(x3)
        x5 = self.stage6(x4)
        x6 = self.stage7(x5)
        x7 = self.conv1(x6)
        x8 = F.adaptive_avg_pool2d(x7, 1)
        x9 = self.conv2(x8)
        y = x9.view(x9.size(0), -1)

        return x1, x2, x3, x4, x5, y

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def get_mobilenetv2(mode, size, cates):
    return MobileNetV2(mode, size, cates)

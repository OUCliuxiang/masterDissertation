"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, mode='t', size=32, cates=100):
        super().__init__()
        self.in_channel = 64
        
        if mode == 't':
            num_block = [2,2,4,4,4]
        else:
            num_block = [1,1,2,2,2]
            
        
        if size == 64 or size == 224:
            self.conv0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)) # 32 / 112
        else:
            self.conv0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)) # 32
        
        if size == 64 or size == 32:
            self.conv1 = self._make_layer(128, num_block[0], True)  # 16
            self.conv2 = self._make_layer(128, num_block[1], True) # 8
            self.conv3 = self._make_layer(256, num_block[2], True) # 4
            self.conv4 = self._make_layer(256, num_block[3], False) # 4
            self.conv5 = self._make_layer(512, num_block[4], False) # 4
        else:
            self.conv1 = self._make_layer(128, num_block[0], True)  # 56
            self.conv2 = self._make_layer(128, num_block[1], True) # 28
            self.conv3 = self._make_layer(256, num_block[2], True) # 14
            self.conv4 = self._make_layer(256, num_block[3], True) # 7
            self.conv5 = self._make_layer(512, num_block[4], False) # 7
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))   

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, cates)
        )
    
    def _make_layer(self, out_channel, num_block, is_pool):
        strides = [1] * (num_block)
        layers = []
        for i in strides:
            layers.append(nn.Conv2d(self.in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            self.in_channel = out_channel
        if is_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.avg_pool(x5)
        x7 = x6.view(x6.size(0), -1)
        y  = self.classifier(x7)

        return x1, x2, x3, x4, x5, y


def get_vgg(mode, size, cates):
    return VGG(mode, size, cates)
import torch.nn as nn


class AuxiliaryClassifier(nn.Module):
    def __init__(self, channel, size, cates):
        super().__init__()
        self.in_channel = channel
        self.size = size
        self.cates = cates
        if size == 7 or size == 4:
            self.strides = [1, 1, 1]
        elif size == 14 or size == 8:
            self.strides = [2, 1, 1]
        elif size == 28 or size == 16:
            self.strides = [2, 2, 1]
        else: # size == 56
            self.strides = [2, 2, 2]

        self.encoder = nn.Sequential(
            # param [inchannel, outchannel, kernel, stride, padding]
            nn.Conv2d(self.in_channel, 256, 3, self.strides[0], 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, self.strides[1], 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, self.strides[2], 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(1024, self.cates)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.linear(x1)
        y  = self.classifier(x2)
        return x2, y



class AutoEncoder(nn.Module):
    def __init__(self, channel, size, cates):
        super().__init__()
        self.in_channel = channel
        self.size = size
        self.cates = cates
        
        if size%7 == 0:
            self.final_length = 7*7*256
            self.final_size = 7
        else:
            self.final_length = 4*4*256
            self.final_size = 4

        if size == 7 or size == 4:
            self.strides = [1, 1, 1]
        elif size == 14 or size == 8:
            self.strides = [2, 1, 1]
        elif size == 28 or size == 16:
            self.strides = [2, 2, 1]
        else: # size == 56
            self.strides = [2, 2, 2]
    
        self.encoder_cnn = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(self.in_channel, 256, 3, self.strides[0], 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, self.strides[1], 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, self.strides[2], 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # self.flatten = nn.flatten(start_dim=1)
        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(self.final_length, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 1024)
        # )

        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(1024, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, self.final_length),
        #     nn.ReLU(inplace=True)
        # )
        # self.unflatten = nn.Unflatten(dim=1, unflattened_size(256, self.final_size, self.final_size))
        self.decoder_cnn = nn.Sequential(
            # param 
            nn.ConvTranspose(256, 256, 3, self.strides[2], 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose(256, 256, 3, self.strides[1], 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose(256, self.in_channel, 3, self.strides[0], 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.in_channel),
            nn.Sigmoid()
        )
        # decoder
        
    
    def forward(self, x):
        # enc = self.encoder_lin(self.flatten(self.encoder_cnn(x)))
        # dec = self.decoder_cnn(self.unflatten(self.decoder_lin(enc)))
        enc = self.encoder_cnn(x)
        dec = self.decoder_cnn(enc)
        return enc, dec
        

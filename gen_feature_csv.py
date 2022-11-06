# -*- coding:utf-8 -*-
# @time :2019.03.15
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import torch
import os
from PIL import Image
# import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
from torchvision import transforms
import csv


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def get_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


if __name__ == "__main__":
    root = "/home/ailven/yingtai/data/"
    # trained_model = cfg.TRAINED_MODEL
    trained_model = root + "/weights/mobilenetv2/cls_c12_320.pth"
    model = load_checkpoint(trained_model)
    print('..... Finished loading model! ......')
    
    model_name = cfg.model_name
    data_path = root + "untrained/test/"
    # data_path = root + "train/"

    name = []
    name.append("cate")
    name.extend([i for i in range(1, 321)])
    
    csvfile = open("cls_untrain_feature.csv", "w+")
    writer = csv.writer(csvfile)
    writer.writerow(name)

    for cate in os.listdir(data_path):
        imgs = os.listdir(os.path.join(data_path, cate))

        ##将模型放置在gpu上运行
        if torch.cuda.is_available():
            model.cuda()
        for i in tqdm(range(len(imgs))):
            img_path = os.path.join(data_path, cate, imgs[i])
            # print(img_path)
            # _id.append(os.path.basename(img_path).split('.')[0])
            img = Image.open(img_path).convert('RGB')
            # print(type(img))
            img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

            line = []
            line.append(cate)
            
            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                fea, out = model(img)

            fea = fea.cpu().numpy().tolist()[0]
            line.extend(fea)
            writer.writerow(line)
    csvfile.close()
                





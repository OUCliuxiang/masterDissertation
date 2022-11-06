#!/usr/bin/env python
# encoding: utf-8

########################################################################
#	Copyright © 2020 OUC_LiuX, All rights reserved. 
#	File Name: dataReinformance.py
#	Author: OUC_LiuX 	Mail:liuxiang@stu.ouc.edu.cn 
#	Version: V1.0.0 
#	Date: 2020-12-07 Mon 22:40
#	Description: 
#	History: 
########################################################################

import sys
import os
import os.path as path
import numpy as np
import cv2
from shutil import copyfile
import numba as nb
import concurrent.futures


@nb.jit()
def pepperSalt(imgIn):
    imgOut = np.zeros(imgIn.shape, np.uint8)
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            for k in range(imgIn.shape[2]):
                rdn = np.random.random()
                if rdn < 0.025:
                    imgOut[i][j][k] = 0
                elif rdn > 0.975:
                    imgOut[i][j][k] = 255
                else:
                    imgOut[i][j][k] = imgIn[i][j][k]
    return imgOut

@nb.jit()
def addGauss(imgIn):
    imgOut = np.array(imgIn/255., dtype=np.float)
    gauss = np.random.normal(imgOut.mean(), imgOut.std(), imgOut.shape)
    imgOut = imgOut*0.9 + gauss*0.1
    imgOut = np.clip(imgOut, 0.0, 1.0)
    imgOut = np.uint8(imgOut*255)
    return imgOut

@nb.jit()
def mixGauss(imgIn):
    imgArray = np.array(imgIn/255., dtype=np.float)
    imgOut = np.zeros(imgArray.shape, np.uint8)
    gauss_noise = np.random.normal(imgArray.mean(), imgArray.std(), imgArray.shape)
    gauss_noise = np.clip( gauss_noise, 0., 1.)
    gauss_noise = np.uint8(gauss_noise*255) 
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            for k in range(imgIn.shape[2]):
                rdn = np.random.random()
                if rdn < 0.05:
                    imgOut[i][j][k] = gauss_noise[i][j][k]
                else:
                    imgOut[i][j][k] = imgIn[i][j][k]
    return imgOut


def addNoise(*x):
    cate_path, img = x
    _img = img.split(".")
    imgIn = cv2.imread(path.join(cate_path, img))
    mode = np.random.randint(0, 3)
    if mode == 0:       # pepperSalt
        img_new = pepperSalt(imgIn)
        cv2.imwrite(path.join(cate_path, _img[0]+"_ps."+_img[1]), img_new)
        return "{} --> {}".format(img, _img[0]+"_ps."+_img[1])

    elif mode == 1:     # addGauss
        img_new = addGauss(imgIn)
        cv2.imwrite(path.join(cate_path, _img[0]+"_addGauss."+_img[1]), img_new)
        return "{} --> {}".format(img, _img[0]+"_addGauss."+_img[1])

    else:               # mixGauss
        img_new = mixGauss(imgIn)
        cv2.imwrite(path.join(cate_path, _img[0]+"_mixGauss."+_img[1]), img_new)
        return "{} --> {}".format(img, _img[0]+"_mixGauss."+_img[1])
    return "Failed!"

if __name__ == "__main__":
    datasets = ["caltech-101", "caltech-256", "cifar-10", "cifar-100", "flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]
    for dataset in datasets:
        train_path = path.join(dataset, "train")
        for cate in os.listdir(train_path):
            cate_path = path.join(train_path, cate)
            img_list = os.listdir(cate_path)
            arg_list = []
            for img in img_list:
                arg_list.append((cate_path, img))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for img, new_img_info in zip(img_list, executor.map(addNoise, arg_list)):
                    print(new_img_info)
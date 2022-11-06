# -*- coding:utf-8 -*-
# Cppyright: OUC_LiuX @ 10Oct2022

import torch
import argparse
# import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_dataset, adjust_learning_rate_cosine, get_net, setup_seed
# import time
from tqdm import tqdm

# 过拟合严重，试一试预训练模型
# 解决了部分情况下的过拟合，imagenet两个数据集还是过拟合，保留问题
# import torchvision.models as models
# mobilenet = models.mobilenet_v2(pretrained=False)
# print(mobilenet)
# num_fits = mobilenet.classifier[1].in_features
# mobilenet.fc = nn.Linear(num_fits, 102)  # 替换最后一个全连接层

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
    except:
        return False, None
    
    model = checkpoint['model'] 
    model.load_state_dict(checkpoint['model_state_dict']) 
    print("== Pretrained model load from {} ".format(filepath))
    return True, model



def train(train_set, val_set, netname, dataset, model, gpus, _lr, _bs, epochs):
    if type(gpus) == int:
        gpus = [gpus]
    if len(gpus) > 1:
        print('****** using multiple gpus to training ********')
        model = nn.DataParallel(model, device_ids=gpus)
        model.cuda(gpus[0])
    else:
        print('****** using single gpu to training ********')
        model.cuda(gpus[0])
    
    # optimizer = optim.SGD(model.parameters(), lr=_lr,
    #             momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-4)
    criterion = nn.CrossEntropyLoss()

    train_iter = DataLoader(train_set, batch_size=_bs, num_workers=16, shuffle=True)
    val_iter = DataLoader(val_set, batch_size = _bs, num_workers=16)

    warmup_epoch = 5
    warmup_steps = warmup_epoch * len(train_iter) / _bs


    train_acc_record = []
    train_loss_record = []
    val_acc_record = []
    val_loss_record = []
    
    steps = 0
    max_acc = 0.0
    max_acc_epoch = 0
    for epoch in range(epochs):
        model.train()
        # start = time.time()
        train_loss = 0.0
        train_acc = 0.0
        total_iter_train = 0
        total_pic_train = 0

        val_loss = 0.0
        val_acc = 0.0
        total_iter_val = 0
        total_pic_val = 0

        with tqdm(total=len(train_iter)) as t:
            for X, y in train_iter:
                X, y = X.cuda(gpus[0]), y.cuda(gpus[0])
                optimizer.zero_grad()
                x1, x2, x3, x4, x5, out = model(X)
                # out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                t.set_description("Epoch [%i/%i] " % (epoch, epochs)) 
                train_loss += loss.cpu().item()
                train_acc += (out.argmax(dim=1) == y).sum().cpu().item()
                total_pic_train += y.shape[0]
                total_iter_train += 1
                steps += 1
                # lr = adjust_learning_rate_cosine(optimizer, 
                #                                  global_step=steps, 
                #                                  learning_rate_base=_lr, 
                #                                  total_steps=len(train_iter) * epochs, 
                #                                  warmup_steps=warmup_steps)
                t.set_postfix(acc=train_acc/total_pic_train, loss=train_loss/total_iter_train)
                t.update()
        
        
        # Test below. 
        # Ssing an enbeded code frame but not another function 
        # for reducing the parameters transmission
        with torch.no_grad():
            model.eval()

            with tqdm(total=len(val_iter)) as t:
                for X, y in val_iter:
                    X, y = X.cuda(gpus[0]), y.cuda(gpus[0])
                    x1, x2, x3, x4, x5, out = model(X)
                    # out = model(X)
                    loss = criterion(out, y)
                    t.set_description("Val ")
                    val_loss += loss.cpu().item()
                    val_acc += (out.argmax(dim=1) == y).sum().cpu().item()
                    total_pic_val += y.shape[0]
                    total_iter_val += 1
                    t.set_postfix(acc=val_acc/total_pic_val, loss=val_loss/total_iter_val)
                    t.update()
            
            if val_acc/total_pic_val - max_acc > 1e-5:
                if len(gpus) > 1:
                    checkpoint = {'model': model.module,
                                'model_state_dict': model.module.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, '../weights/baseline/{}_{}_best.pth'.format(netname, dataset, epoch))
                else:
                    checkpoint = {'model': model,
                                'model_state_dict': model.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, '../weights/baseline/{}_{}_best.pth'.format(netname, dataset, epoch))
                max_acc = val_acc/total_pic_val
                max_acc_epoch = epoch
                print("best model {} for {} saved @ epoch {} with acc = {}".format(netname, dataset, epoch, val_acc/total_pic_val) )
        
        train_acc_record.append(train_acc/total_pic_train)
        train_loss_record.append(train_loss/total_iter_train)
        val_acc_record.append(val_acc/total_pic_val)
        val_loss_record.append(val_loss/total_iter_val)

        if epoch - max_acc_epoch >= 10:
            break
    
    np.save("./data/baseline/{}_{}_train_acc.npy".format(netname, dataset), train_acc_record)
    np.save("./data/baseline/{}_{}_train_loss.npy".format(netname, dataset), train_loss_record)
    np.save("./data/baseline/{}_{}_val_acc.npy".format(netname, dataset), val_acc_record)
    np.save("./data/baseline/{}_{}_val_loss.npy".format(netname, dataset), val_loss_record)

    if len(gpus) > 1:
        checkpoint = {'model': model.module,
                        'model_state_dict': model.module.state_dict(),
                        'epoch': epoch}
        torch.save(checkpoint, '../weights/baseline/{}_{}_last.pth'.format(netname, dataset, epoch))
    else:
        checkpoint = {'model': model,
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch}
        torch.save(checkpoint, '../weights/baseline/{}_{}_last.pth'.format(netname, dataset, epoch))
    # print("best model {} for {} saved @ epoch {} with acc = {}".format(netname, dataset, epoch, val_acc/total_pic_val) )
    with open("./train_record.txt", "a+") as f:
        f.write("{}_{}_baseline stop @ {} with {}\n".format(netname, dataset, epoch, val_acc/total_pic_val))


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("[ERROR] Can only run on GPUs\n")
        exit(1)

    # 设置随机数种子
    setup_seed(42)

    datasets = ["caltech-101", "caltech-256", "cifar-10", "cifar-100", 
                "flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]
    # datasets = ["imagenet-mini", "svhn", "tiny-imagenet"]
    netnames = ["vgg-t", "vgg-s", "mobilenet-t", "mobilenet-s", 
                "resnet-t", "resnet-s"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="All", 
                        help="point a specific dataset or just empty this arg so use all dataset")
    parser.add_argument("--net", type=str, default="All", 
                        help = "select a net from [\"vgg-t\", \"vgg-s\", \"mobilenet-t\", \"mobilenet-s\", \"resnet-t\", \"resnet-s\"]")
    parser.add_argument("--resume", type=str, default=None, help="path of resumed model")
    parser.add_argument("--gpus", type=int, nargs='+', default=0, help="how many gpus used in this training precedure")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="max epochs, but it will auto stop while network converged.")
    args = parser.parse_args()

    if args.dataset == "All" and args.net == "All":
        for dataset in datasets:
            ret, train_set, val_set = get_dataset(dataset)
            if not ret:
                print("dataset {} not found".format(dataset))
                exit(1)
            
            print("== dataset {} done ==".format(dataset) )
    
            for netname in netnames:
                if args.resume:
                    print("== When args.net is All, args.reusme is forbidden")
                    exit(1)
                else:
                    print("== Initing model: {}".format(netname))
                    ret, model = get_net(dataset, netname)
                
                if not ret:
                    print("network {} not found".format(netname))
                    exit(1)
                
                for _, p in model.named_parameters():
                    p.requires_grad = True
                
                train(train_set, val_set, netname, dataset, model, args.gpus, args.lr, args.bs, args.epochs)
    
    elif args.net == "All":
        if not args.dataset in datasets:
            print("Fdataset {} not found".format(args.dataset))
            exit(1)
        ret, train_set, val_set = get_dataset(args.dataset) 
        if not ret:
            print("Failed obtain dataset {}".format(args.dataset))
            exit(1)
        print("== dataset {} done ==".format(args.dataset) )

        for netname in netnames:
            if args.resume:
                print("== When args.net is All, args.reusme is forbidden")
                exit(1)
            else:
                print("== Initing model: {} ==".format(netname))
                ret, model = get_net(args.dataset, netname)
            
            if not ret:
                print("Failed load network {}".format(netname))
                exit(1)
            
            for _, p in model.named_parameters():
                p.requires_grad = True
            
            train(train_set, val_set, netname, args.dataset, model, args.gpus, args.lr, args.bs, args.epochs)
    
    elif args.dataset == "All":
        if not args.net in netnames:
            print("net {} not found".format(args.net))
            exit(1)
        
        for dataset in datasets:
            ret, train_set, val_set = get_dataset(dataset)
            if not ret:
                print("Failed obtain dataset {}".format(dataset))
                exit(1)
            print("== dataset {} done ==".format(dataset) )

            if args.resume:
                print("== Resuming model from {} ".format(args.resume))
                ret, model = load_checkpoint(args.resume)
            else:
                print("== Initing model: {}".format(args.net))
                ret, model = get_net(dataset, args.net)
            
            if not ret:
                print("Failed load network {}".format(args.net))
                exit(1)
            
            for _, p in model.named_parameters():
                p.requires_grad = True
            train(train_set, val_set, args.net, dataset, model, args.gpus, args.lr, args.bs, args.epochs)
    
    else: # None is All
        if not args.net in netnames:
            print("net {} not found".format(args.net))
            exit(1)
        
        if not args.dataset in datasets:
            print("dataset {} not found".format(args.dataset))
            exit(1)
        
        ret, train_set, val_set = get_dataset(args.dataset)
        print("== dataset {} done ==".format(args.dataset) )

        if not ret:
            print("Failed obtain dataset {}".format(args.dataset))
            exit(1)
        
        if args.resume:
            print("== Resuming model from {} ".format(args.resume))
            ret, model = load_checkpoint(args.resume)
        else:
            print("== Initing model: {}".format(args.net))
            ret, model = get_net(args.dataset, args.net)
        
        if not ret:
            print("Failed load network {}".format(args.net))
            exit(1)
        
        for _, p in model.named_parameters():
            p.requires_grad = True

        train(train_set, val_set, args.net, args.dataset, model, args.gpus, args.lr, args.bs, args.epochs)
        # train(train_set, val_set, args.net, args.dataset, mobilenet, args.gpus, args.lr, args.bs, args.epochs)

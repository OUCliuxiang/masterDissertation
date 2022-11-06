# -*- coding:utf-8 -*-
# Cppyright: OUC_LiuX @ 10Oct2022

import torch
import argparse
# import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_dataset, adjust_learning_rate_cosine, get_net, setup_seed
# import time
from tqdm import tqdm

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# channel wise distillation 
def loss_kl(teacher, student, T):
    # shape: B C H W
    B, C, H, W = teacher.
    C = C * 1.0
    teacher = torch.flatten(teacher, start_dim=2)
    student = torch.flatten(student, start_dim=2)
    return (T * T / C) * nn.KLDivLoss(reducion="batchmean")(
                        F.softmax(student/T, dim=2), F.softmax(teacher/T, dim=2))

def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
    KD_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
    except:
        print("{} load failed.".format(filepath))
        return False, None
    
    model = checkpoint['model'] 
    model.load_state_dict(checkpoint['model_state_dict']) 
    print("== Pretrained model load from {} ".format(filepath))
    return True, model



def train(train_set, val_set, netname, dataset, model_t, model_s, gpus, _lr, _bs, epochs, T, alpha):
    
    if dataset == "caltech-101":
        size = [56, 28, 14, 7, 7]
        cates = 102
    elif dataset == "caltech-256":
        size = [56, 28, 14, 7, 7]
        cates = 257
    elif dataset == "cifar-10":
        size = [16, 8, 4, 4, 4]
        cates = 10
    elif dataset == "cifar-100":
        size = [16, 8, 4, 4, 4]
        cates = 100
    elif dataset == "flower-102":
        size = [56, 28, 14, 7, 7]
        cates = 102
    elif dataset == "imagenet-mini":
        size = [56, 28, 14, 7, 7]
        cates = 1000
    elif dataset == "svhn":
        size = [16, 8, 4, 4, 4]
        cates = 10
    elif dataset == "tiny-imagenet":
        size = [16, 8, 4, 4, 4]
        cates = 200
    else:
        print("wrong dataset path: {}".format(dataset))
        exit(1)
    
    if "vgg" in netname:
        channel = [64, 128, 256, 512, 512]
    elif "resnet" in netname:
        channel = [64, 64, 128, 256, 512]
    elif "mobilenet" in netname:
        channel = [24, 32, 64, 96, 160]
    else:
        print("net name not found: {}".format(netname))
    

    from models.encoder import AuxiliaryClassifier
    # param channel, size, cates
    AC1 = AuxiliaryClassifier(channel[0], size[0], cates)
    AC2 = AuxiliaryClassifier(channel[1], size[1], cates)
    AC3 = AuxiliaryClassifier(channel[2], size[2], cates)
    AC4 = AuxiliaryClassifier(channel[3], size[3], cates)
    AC5 = AuxiliaryClassifier(channel[4], size[4], cates)
    
    if type(gpus) == int:
        gpus = [gpus]
    if len(gpus) > 1:
        print('****** using multiple gpus to training ********')
        model_t = nn.DataParallel(model_t, device_ids=gpus)
        model_t.cuda(gpus[0])
        model_s = nn.DataParallel(model_s, device_ids=gpus)
        model_s.cuda(gpus[0])
        AC1 = nn.DataParallel(AC1, device_ids=gpus)
        AC1.cuda(gpus[0])
        AC2 = nn.DataParallel(AC2, device_ids=gpus)
        AC2.cuda(gpus[0])
        AC3 = nn.DataParallel(AC3, device_ids=gpus)
        AC3.cuda(gpus[0])
        AC4 = nn.DataParallel(AC4, device_ids=gpus)
        AC4.cuda(gpus[0])
        AC5 = nn.DataParallel(AC5, device_ids=gpus)
        AC5.cuda(gpus[0])

    else:
        print('****** using single gpu:{} to training ********'.format(gpus[0]))
        model_t.cuda(gpus[0])
        model_s.cuda(gpus[0])
        AC1.cuda(gpus[0])
        AC2.cuda(gpus[0])
        AC3.cuda(gpus[0])
        AC4.cuda(gpus[0])
        AC5.cuda(gpus[0])

    opt1 = optim.SGD(AC1.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri1 = nn.CrossEntropyLoss()
    
    opt2 = optim.SGD(AC2.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri2 = nn.CrossEntropyLoss()

    opt3 = optim.SGD(AC3.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri3 = nn.CrossEntropyLoss()

    opt4 = optim.SGD(AC4.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri4 = nn.CrossEntropyLoss()

    opt5 = optim.SGD(AC5.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri5 = nn.CrossEntropyLoss()

    opt_t = optim.Adam(filter(lambda p: p.requires_grad, model_t.parameters()), lr = 3e-4)
    cri_t = nn.CrossEntropyLoss()
    
    opt_s = optim.Adam(filter(lambda p: p.requires_grad, model_s.parameters()), lr = 3e-4)
    cri_s = nn.CrossEntropyLoss()
    
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
        
        model_s.train()

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

                for _, p in model_t.named_parameters():
                    p.requires_grad = True
                for _, p in AC1.named_parameters():
                    p.requires_grad = True
                for _, p in AC2.named_parameters():
                    p.requires_grad = True
                for _, p in AC3.named_parameters():
                    p.requires_grad = True
                for _, p in AC4.named_parameters():
                    p.requires_grad = True
                for _, p in AC5.named_parameters():
                    p.requires_grad = True

                model_t.train()
                AC1.train()
                AC2.train()
                AC3.train()
                AC4.train()
                AC5.train()

                opt_t.zero_grad()
                opt1.zero_grad()
                opt2.zero_grad()
                opt3.zero_grad()
                opt4.zero_grad()
                opt5.zero_grad()

                t1, t2, t3, t4, t5, teacher_out = model_t(X)

                a1, y1 = AC1(t1.detach())
                loss1 = cri1(y1, y)
                loss1.backward()
                opt1.step()
                for _, p in AC1.named_parameters():
                    p.requires_grad = False
                AC1.eval()        

                a2, y2 = AC2(t2.detach())
                loss2 = cri2(y2, y)
                loss2.backward()
                opt2.step()
                for _, p in AC2.named_parameters():
                    p.requires_grad = False
                AC2.eval()
                
                a3, y3 = AC3(t3.detach())
                loss3 = cri3(y3, y)
                loss3.backward()
                opt3.step()
                for _, p in AC3.named_parameters():
                    p.requires_grad = False
                AC3.eval()
                
                a4, y4 = AC4(t4.detach())
                loss4 = cri4(y4, y)
                loss4.backward()
                opt4.step()
                for _, p in AC4.named_parameters():
                    p.requires_grad = False
                AC4.eval()
                
                a5, y5 = AC5(t5.detach())
                loss5 = cri5(y5, y)
                loss5.backward()
                opt5.step()
                for _, p in AC5.named_parameters():
                    p.requires_grad = False
                AC5.eval()

                loss_t = cri_t(teacher_out, y)
                loss_t.backward()
                opt_t.step()
                for _, p in model_t.named_parameters():
                    p.requires_grad = False
                model_t.eval()

                opt_s.zero_grad()
                x1, x2, x3, x4, x5, out = model_s(X)
                t1, t2, t3, t4, t5, teacher_out = model_t(X)
                loss = 0.5 * loss_fn_kd(out, y, teacher_out, T, alpha) \
                     + 0.1 * loss_kl(AC1(t1)[0], AC1(x1)[0], T) \
                     + 0.1 * loss_kl(AC2(t2)[0], AC2(x2)[0], T) \
                     + 0.1 * loss_kl(AC3(t3)[0], AC3(x3)[0], T) \
                     + 0.1 * loss_kl(AC4(t4)[0], AC4(x4)[0], T) \
                     + 0.1 * loss_kl(AC5(t5)[0], AC5(x5)[0], T)
                loss.backward()
                opt_s.step()
                t.set_description("Epoch [%i/%i] " % (epoch, epochs)) 
                train_loss += loss.cpu().item()
                train_acc += (out.argmax(dim=1) == y).sum().cpu().item()
                total_pic_train += y.shape[0]
                total_iter_train += 1

                steps += 1
                lr = adjust_learning_rate_cosine(opt1, 
                                                 global_step=steps, 
                                                 learning_rate_base=_lr, 
                                                 total_steps=len(train_iter) * epochs, 
                                                 warmup_steps=warmup_steps)
                lr = adjust_learning_rate_cosine(opt2, 
                                                 global_step=steps, 
                                                 learning_rate_base=_lr, 
                                                 total_steps=len(train_iter) * epochs, 
                                                 warmup_steps=warmup_steps)
                lr = adjust_learning_rate_cosine(opt3, 
                                                 global_step=steps, 
                                                 learning_rate_base=_lr, 
                                                 total_steps=len(train_iter) * epochs, 
                                                 warmup_steps=warmup_steps)                            
                lr = adjust_learning_rate_cosine(opt4, 
                                                 global_step=steps, 
                                                 learning_rate_base=_lr, 
                                                 total_steps=len(train_iter) * epochs, 
                                                 warmup_steps=warmup_steps)
                lr = adjust_learning_rate_cosine(opt5, 
                                                 global_step=steps, 
                                                 learning_rate_base=_lr, 
                                                 total_steps=len(train_iter) * epochs, 
                                                 warmup_steps=warmup_steps)
                
                t.set_postfix(acc=train_acc/total_pic_train, loss=train_loss/total_iter_train)
                t.update()

        with torch.no_grad():
            model_s.eval()

            with tqdm(total=len(val_iter)) as t:
                for X, y in val_iter:
                    X, y = X.cuda(gpus[0]), y.cuda(gpus[0])
                    x1, x2, x3, x4, x5, out = model_s(X)
                    loss = cri_s(out, y)
                    t.set_description("Val ")
                    val_loss += loss.cpu().item()
                    val_acc += (out.argmax(dim=1) == y).sum().cpu().item()
                    total_pic_val += y.shape[0]
                    total_iter_val += 1
                    t.set_postfix(acc=val_acc/total_pic_val, loss=val_loss/total_iter_val)
                    t.update()
            
            if val_acc/total_pic_val - max_acc > 1e-5:
                if len(gpus) > 1:
                    checkpoint = {'model': model_s.module,
                                'model_state_dict': model_s.module.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, '../weights/kd_ac_online_syn/{}_{}_best.pth'.format(netname, dataset, epoch))
                else:
                    checkpoint = {'model': model_s,
                                'model_state_dict': model_s.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, '../weights/kd_ac_online_syn/{}_{}_best.pth'.format(netname, dataset, epoch))
                max_acc = val_acc/total_pic_val
                max_acc_epoch = epoch
                print("best model {} for {} saved @ epoch {} with acc = {}".format(netname, dataset, epoch, val_acc/total_pic_val) )
        
        train_acc_record.append(train_acc/total_pic_train)
        train_loss_record.append(train_loss/total_iter_train)
        val_acc_record.append(val_acc/total_pic_val)
        val_loss_record.append(val_loss/total_iter_val)

        if epoch - max_acc_epoch > 10:
            break
    
    np.save("./data/kd_ac_online_syn/{}_{}_train_acc.npy".format(netname, dataset), train_acc_record)
    np.save("./data/kd_ac_online_syn/{}_{}_train_loss.npy".format(netname, dataset), train_loss_record)
    np.save("./data/kd_ac_online_syn/{}_{}_val_acc.npy".format(netname, dataset), val_acc_record)
    np.save("./data/kd_ac_online_syn/{}_{}_val_loss.npy".format(netname, dataset), val_loss_record)

    if len(gpus) > 1:
        checkpoint = {'model': model_s.module,
                        'model_state_dict': model_s.module.state_dict(),
                        'epoch': epoch}
        torch.save(checkpoint, '../weights/kd_ac_online_syn/{}_{}_last.pth'.format(netname, dataset, epoch))
    else:
        checkpoint = {'model': model_s,
                    'model_state_dict': model_s.state_dict(),
                    'epoch': epoch}
        torch.save(checkpoint, '../weights/kd_ac_online_syn/{}_{}_last.pth'.format(netname, dataset, epoch))
    # print("best model {} for {} saved @ epoch {} with acc = {}".format(netname, dataset, epoch, val_acc/total_pic_val) )
    with open("./train_record.txt", "a+") as f:
        f.write("{}_{}_ac_online_syn stop @ {} with {}\n".format(netname, dataset, epoch, val_acc/total_pic_val))

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("[ERROR] Can only run on GPUs\n")
        exit(1)

    # 设置随机数种子
    setup_seed(42)

    datasets = ["caltech-101", "caltech-256", "cifar-10", "cifar-100", 
                "flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]
    # datasets = ["flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]
    netnames = ["resnet", "vgg", "mobilenet"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="All", 
                        help="point a specific dataset or just empty this arg so use all dataset")
    parser.add_argument("--net", type=str, default="All", 
                        help = "select a net from [\"vgg\", \"mobilenet\", \"resnet\"")
    # parser.add_argument("--resume", type=str, default=None, help="path of resumed model") # kd mode does not need resume
    parser.add_argument("--gpus", type=int, nargs='+', default=0, help="how many gpus used in this training precedure")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--T", type=int, default=10, help="Tempture of distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="ratio of distillation loss [0--1]")
    args = parser.parse_args()

    if args.dataset == "All" and args.net == "All":
        for dataset in datasets:
            ret, train_set, val_set = get_dataset(dataset)
            if not ret:
                print("dataset {} not found".format(dataset))
                exit(1)
            
            print("== dataset {} done ==".format(dataset) )
    
            for netname in netnames:
                print("== Initing student model: {}-s".format(netname))
                ret, model_s = get_net(dataset, netname+"-s")
                if not ret:
                    print("network {}-s not found".format(netname))
                    exit(1)

                print("== Initing teacher model: {}-t".format(netname))
                ret, model_t = get_net(dataset, netname+"-t")
                if not ret:
                    print("network {}-t not found".format(netname))
                    exit(1)
                
                for _, p in model_t.named_parameters():
                    p.requires_grad = True

                for _, p in model_s.named_parameters():
                    p.requires_grad = True
                
                
                train(train_set, val_set, netname, dataset, model_t, model_s, args.gpus, args.lr, args.bs, args.epochs, args.T, args.alpha)
    
    elif args.net == "All":
        if not args.dataset in datasets:
            print("dataset {} not found".format(args.dataset))
            exit(1)
        ret, train_set, val_set = get_dataset(args.dataset) 
        if not ret:
            print("Failed obtain dataset {}".format(args.dataset))
            exit(1)
        print("== dataset {} done ==".format(args.dataset) )

        for netname in netnames:
            print("== Initing student model: {} ==".format(netname))
            ret, model_s = get_net(args.dataset, netname+"-s")
            if not ret:
                print("Failed load network {}".format(netname))
                exit(1)
            
            print("== Loading teacher model: {}".format(netname))
            ret, model_t = load_checkpoint("../weights/baseline/{}-t_{}_best.pth".format(netname, args.dataset))
            if not ret:
                print("teacher model {} load failed".format(netname))
                exit(1)

            for _, p in model_s.named_parameters():
                p.requires_grad = True
            for _, p in model_t.named_parameters():
                p.requires_grad = True

            train(train_set, val_set, netname, args.dataset, model_t, model_s, args.gpus, args.lr, args.bs, args.epochs, args.T, args.alpha)
    
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

            print("== Initing student model: {}".format(args.net))
            ret, model_s = get_net(dataset, args.net+"-s")
            if not ret:
                print("Failed load network {}".format(args.net))
                exit(1)
            
            print("== Loading teacher model: {}".format(args.net))
            ret, model_t = load_checkpoint("../weights/baseline/{}-t_{}_best.pth".format(args.net, args.dataset))
            if not ret:
                print("teacher model {} load failed".format(args.net))
                exit(1)
            
            for _, p in model_s.named_parameters():
                p.requires_grad = True
            for _, p in model_t.named_parameters():
                p.requires_grad = False

            train(train_set, val_set, args.net, args.dataset, model_t, model_s, args.gpus, args.lr, args.bs, args.epochs, args.T, args.alpha)
    
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
        
        print("== Initing model: {}".format(args.net))
        ret, model_s = get_net(args.dataset, args.net+"-s")
        if not ret:
            print("Failed load network {}".format(args.net))
            exit(1)

        print("== Loading teacher model: {}".format(args.net))
        ret, model_t = load_checkpoint("../weights/baseline/{}-t_{}_best.pth".format(args.net, args.dataset))
        if not ret:
            print("teacher model {} load failed".format(args.net))
            exit(1)


        for _, p in model_s.named_parameters():
            p.requires_grad = True
        for _, p in model_t.named_parameters():
            p.requires_grad = False

        train(train_set, val_set, args.net, args.dataset, model_t, model_s, args.gpus, args.lr, args.bs, args.epochs, args.T, args.alpha)

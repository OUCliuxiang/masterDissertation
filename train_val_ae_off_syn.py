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
    

    from models.encoder import AutoEncoder
    # param channel, size, cates
    AE1 = AutoEncoder(channel[0], size[0], cates)
    AE2 = AutoEncoder(channel[1], size[1], cates)
    AE3 = AutoEncoder(channel[2], size[2], cates)
    AE4 = AutoEncoder(channel[3], size[3], cates)
    AE5 = AutoEncoder(channel[4], size[4], cates)

    for _, p in AE1.named_parameters():
        p.requires_grad = True
    for _, p in AE2.named_parameters():
        p.requires_grad = True
    for _, p in AE3.named_parameters():
        p.requires_grad = True
    for _, p in AE4.named_parameters():
        p.requires_grad = True
    for _, p in AE5.named_parameters():
        p.requires_grad = True
    
    if type(gpus) == int:
        gpus = [gpus]
    if len(gpus) > 1:
        print('****** using multiple gpus to training ********')
        model_t = nn.DataParallel(model_t, device_ids=gpus)
        model_t.cuda(gpus[0])
        model_s = nn.DataParallel(model_s, device_ids=gpus)
        model_s.cuda(gpus[0])
        AE1 = nn.DataParallel(AE1, device_ids=gpus)
        AE1.cuda(gpus[0])
        AE2 = nn.DataParallel(AE2, device_ids=gpus)
        AE2.cuda(gpus[0])
        AE3 = nn.DataParallel(AE3, device_ids=gpus)
        AE3.cuda(gpus[0])
        AE4 = nn.DataParallel(AE4, device_ids=gpus)
        AE4.cuda(gpus[0])
        AE5 = nn.DataParallel(AE5, device_ids=gpus)
        AE5.cuda(gpus[0])

    else:
        print('****** using single gpu:{} to training ********'.format(gpus[0]))
        model_t.cuda(gpus[0])
        model_s.cuda(gpus[0])
        AE1.cuda(gpus[0])
        AE2.cuda(gpus[0])
        AE3.cuda(gpus[0])
        AE4.cuda(gpus[0])
        AE5.cuda(gpus[0])

    opt1 = optim.SGD(AE1.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri1 = nn.MSELoss()
    
    opt2 = optim.SGD(AE2.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri2 = nn.MSELoss()

    opt3 = optim.SGD(AE3.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri3 = nn.MSELoss()

    opt4 = optim.SGD(AE4.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri4 = nn.MSELoss()

    opt5 = optim.SGD(AE5.parameters(), lr=_lr,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    cri5 = nn.MSELoss()

    # optimizer = optim.SGD(model_s.parameters(), lr=_lr,
    #             momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_s.parameters()), lr = 3e-4)
    criterion = nn.CrossEntropyLoss()

    train_iter = DataLoader(train_set, batch_size=_bs, num_workers=16, shuffle=True)
    val_iter = DataLoader(val_set, batch_size = _bs, num_workers=16)

    warmup_epoch = 5
    warmup_steps = warmup_epoch * len(train_iter) / _bs
    
    
    with torch.no_grad():
        model_t.eval()
        val_loss = 0.0
        val_acc = 0.0
        total_iter_val = 0
        total_pic_val = 0
        with tqdm(total=len(val_iter)) as t:
            for X, y in val_iter:
                X, y = X.cuda(gpus[0]), y.cuda(gpus[0])
                x1, x2, x3, x4, x5, out = model_t(X)
                t.set_description("Teacher @ Val ")
                val_acc += (out.argmax(dim=1) == y).sum().cpu().item()
                total_pic_val += y.shape[0]
                total_iter_val += 1
                t.set_postfix(acc=val_acc/total_pic_val, loss=0)
                t.update()
    
    
    train_acc_record = []
    train_loss_record = []
    val_acc_record = []
    val_loss_record = []

    steps = 0
    max_acc = 0.0
    max_acc_epoch = 0
    for epoch in range(epochs):
        model_s.train()
        model_t.eval()
        AE1.eval()
        AE2.eval()
        AE3.eval()
        AE4.eval()
        AE5.eval()
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

                AE1.train()
                AE2.train()
                AE3.train()
                AE4.train()
                AE5.train()

                for _, p in AE1.named_parameters():
                    p.requires_grad = True
                for _, p in AE2.named_parameters():
                    p.requires_grad = True
                for _, p in AE3.named_parameters():
                    p.requires_grad = True
                for _, p in AE4.named_parameters():
                    p.requires_grad = True
                for _, p in AE5.named_parameters():
                    p.requires_grad = True

                opt1.zero_grad()
                opt2.zero_grad()
                opt3.zero_grad()
                opt4.zero_grad()
                opt5.zero_grad()
                
                t1, t2, t3, t4, t5, teacher_out = model_t(X)
                x1, x2, x3, x4, x5, out = model_s(X)

                t1 = t1.detach()
                t2 = t2.detach()
                t3 = t3.detach()
                t4 = t4.detach()
                t5 = t5.detach()

                a1, y1 = AE1(t1)
                a2, y2 = AE2(t2)
                a3, y3 = AE3(t3)
                a4, y4 = AE4(t4)
                a5, y5 = AE5(t5)

                loss1 = cri1(y1, t1)
                loss1.backward()
                opt1.step()
                
                loss2 = cri2(y2, t2)
                loss2.backward()
                opt2.step()
                
                loss3 = cri3(y3, t3)
                loss3.backward()
                opt3.step()
                
                loss4 = cri4(y4, t4)
                loss4.backward()
                opt4.step()
                
                loss5 = cri5(y5, t5)
                loss5.backward()
                opt5.step()

                AE1.eval()
                AE2.eval()
                AE3.eval()
                AE4.eval()
                AE5.eval()

                for _, p in AE1.named_parameters():
                    p.requires_grad = False
                for _, p in AE2.named_parameters():
                    p.requires_grad = False
                for _, p in AE3.named_parameters():
                    p.requires_grad = False
                for _, p in AE4.named_parameters():
                    p.requires_grad = False
                for _, p in AE5.named_parameters():
                    p.requires_grad = False

                # loss = criterion(out, y)
                loss = 0.5 * loss_fn_kd(out, y, teacher_out, T, alpha) \
                     + 0.1 * loss_kl(AE1(t1)[0], AE1(x1)[0], T) \
                     + 0.1 * loss_kl(AE2(t2)[0], AE2(x2)[0], T) \
                     + 0.1 * loss_kl(AE3(t3)[0], AE3(x3)[0], T) \
                     + 0.1 * loss_kl(AE4(t4)[0], AE4(x4)[0], T) \
                     + 0.1 * loss_kl(AE5(t5)[0], AE5(x5)[0], T)
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
        
        
        # Test below. 
        # Ssing an enbeded code frame but not another function 
        # for reducing the parameters transmission
        with torch.no_grad():
            model_s.eval()

            with tqdm(total=len(val_iter)) as t:
                for X, y in val_iter:
                    X, y = X.cuda(gpus[0]), y.cuda(gpus[0])
                    x1, x2, x3, x4, x5, out = model_s(X)
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
                    checkpoint = {'model': model_s.module,
                                'model_state_dict': model_s.module.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, '../weights/kd_ae_off_syn/{}_{}_best.pth'.format(netname, dataset, epoch))
                else:
                    checkpoint = {'model': model_s,
                                'model_state_dict': model_s.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, '../weights/kd_ae_off_syn/{}_{}_best.pth'.format(netname, dataset, epoch))
                max_acc = val_acc/total_pic_val
                max_acc_epoch = epoch
                print("best model {} for {} saved @ epoch {} with acc = {}".format(netname, dataset, epoch, val_acc/total_pic_val) )
        
        
        train_acc_record.append(train_acc/total_pic_train)
        train_loss_record.append(train_loss/total_iter_train)
        val_acc_record.append(val_acc/total_pic_val)
        val_loss_record.append(val_loss/total_iter_val)

        if epoch - max_acc_epoch > 10:
            break
    
    np.save("./data/kd_ae_off_syn/{}_{}_train_acc.npy".format(netname, dataset), train_acc_record)
    np.save("./data/kd_ae_off_syn/{}_{}_train_loss.npy".format(netname, dataset), train_loss_record)
    np.save("./data/kd_ae_off_syn/{}_{}_val_acc.npy".format(netname, dataset), val_acc_record)
    np.save("./data/kd_ae_off_syn/{}_{}_val_loss.npy".format(netname, dataset), val_loss_record)

    if len(gpus) > 1:
        checkpoint = {'model': model_s.module,
                        'model_state_dict': model_s.module.state_dict(),
                        'epoch': epoch}
        torch.save(checkpoint, '../weights/kd_ae_off_syn/{}_{}_last.pth'.format(netname, dataset, epoch))
    else:
        checkpoint = {'model': model_s,
                    'model_state_dict': model_s.state_dict(),
                    'epoch': epoch}
        torch.save(checkpoint, '../weights/kd_ae_off_syn/{}_{}_last.pth'.format(netname, dataset, epoch))
    # print("best model {} for {} saved @ epoch {} with acc = {}".format(netname, dataset, epoch, val_acc/total_pic_val) )
    with open("./train_record.txt", "a+") as f:
        f.write("{}_{}_ae_off_syn stop @ {} with {}\n".format(netname, dataset, epoch, val_acc/total_pic_val))

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("[ERROR] Can only run on GPUs\n")
        exit(1)

    # ?????????????????????
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
                print("== Initing student model: {}".format(netname))
                ret, model_s = get_net(dataset, netname+"-s")
                if not ret:
                    print("network {} not found".format(netname))
                    exit(1)

                print("== Loading teacher model: {}".format(netname))
                ret, model_t = load_checkpoint("../weights/baseline/{}-t_{}_best.pth".format(netname, dataset))
                if not ret:
                    print("teacher model {} load failed".format(netname))
                    exit(1)
                
                for _, p in model_t.named_parameters():
                    p.requires_grad = False

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
                p.requires_grad = False

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
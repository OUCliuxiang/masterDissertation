import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_dataset
from tqdm import tqdm


def calculate_top_k_accuracy(logits, targets, k=5):
    values, indices = torch.topk(logits, k=k, sorted=True)
    y = torch.reshape(targets, [-1, 1])
    correct = (y == indices) * 1.  # 对比预测的K个值中是否包含有正确标签中的结果
    top_k_accuracy = torch.mean(correct) * k  # 计算最后的准确率
    return top_k_accuracy


def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
    except:
        return False, None
    
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return True, model


def test( val_set, model, mode, gpu, dataset, netname):
    
    val_iter = DataLoader(val_set, batch_size = 256, num_workers=16)

    total_1 = 0
    total_5 = 0
    max_val_len = 0
    iters = 0
    
    for X, y in val_iter:
        X, y = X.cuda(gpu), y.cuda(gpu)
        x1, x2, x3, x4, x5, out = model(X)
        ret, predictions = torch.max(out.data, 1)
        # correct = (predictions == labels).sum()
        top1 = calculate_top_k_accuracy(out, y, 1)
        top5 = calculate_top_k_accuracy(out, y, 5)
        total_1 += top1*y.shape[0]
        total_5 += top5*y.shape[0]
        iters += 1
        max_val_len += y.shape[0]
        # print('Iteration: {}/{}'.format(iters, len(val_iter)), 
        #         'top1: %.3f' % top1.float(), 
        #         'top5: %.3f' % top5.float())
    print('{} @ {}: '.format(netname, dataset),
          'Top1 ACC: %.3f'%(total_1.float()/max_val_len), 
          'Top5 ACC: %.3f'%(total_5.float()/max_val_len))


    # return total_correct.float()/max_val_len



if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("[ERROR] Can only run on GPUs\n")
        exit(1)
    
    datasets = ["caltech-101", "caltech-256", "cifar-10", "cifar-100", 
                "flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]
    # datasets = ["flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="All", 
                        help="point a specific dataset or just empty this arg so use all dataset")
    parser.add_argument("--net", type=str, default="All", 
                        help = "select a net from [\"vgg-t\", \"vgg-s\", \"mobilenet-t\", \"mobilenet-s\", \"resnet-t\", \"resnet-s\"]")
    parser.add_argument("--mode", type=str, default="baseline", 
                        help = "baseline, kd, or others, map into the subpath below ../weights/")
    parser.add_argument("--gpus", type=int, default=0, help="appoint only ONE gpu")
    args = parser.parse_args()
    
    if args.mode == "kd":
        netnames = ["vgg", "mobilenet", "resnet"]
    elif args.mode == "baseline":
        netnames = ["vgg-t", "vgg-s", "mobilenet-t", 
                    "mobilenet-s", "resnet-t", "resnet-s"]
    else:
        print("not support only kd and baseline")
        exit(1)

    if args.dataset == "All" and args.net == "All":
        for dataset in datasets:
            ret, train_set, val_set = get_dataset(dataset)
            if not ret:
                print("dataset {} not found".format(dataset))
                exit(1)
            
            print("== dataset {} done ==".format(dataset) )
    
            for netname in netnames:
                # print("../weights/"+args.mode+"/"+args.net+"_"+dataset+"_best_singlegpu.pth")
                ret, model = load_checkpoint("../weights/"+args.mode+"/"+netname+"_"+dataset+"_best_singlegpu.pth")
                if not ret:
                    print("network {} not found".format(netname))
                    exit(1)
                model.cuda(args.gpus)
                with torch.no_grad():
                    model.eval()
                    test(val_set, model, args.mode, args.gpus, dataset, netname)

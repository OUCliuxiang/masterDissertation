import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import prettytable as pt
from torch.utils.data import DataLoader
from utils import get_dataset
from tqdm import tqdm

NUM_CLASSES = 102

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))#初始化混淆矩阵，元素都为0
        self.num_classes = num_classes#类别数量，本例数据集类别为5
        self.labels = labels#类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):#pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1#根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):#计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]#混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n#总体准确率
        print("the model accuracy is ", acc)
		
		# kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = pt.PrettyTable()#创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        avgp = 0.
        avgr = 0.
        avgs = 0.
        for i in range(self.num_classes):#精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.#每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            avgp += Precision
            avgr += Recall
            avgs += Specificity
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        avgp /= self.num_classes
        avgr /= self.num_classes
        avgs /= self.num_classes
        table.add_row(["Avg", avgp, avgr, avgs])
        print(table)
        return str(acc), avgp, avgr, avgs

    def plot(self, path):#绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.figure(dpi=441)
        plt.figure().set_size_inches(28,21)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()[0]+')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        # plt.tight_layout()
        # plt.show()
        plt.savefig(path, dpi='figure', pad_inches=0.1)


def test(_cates, val_set, model, mode, dataset, netname):
    cates = [None] * _cates
    for i in range(_cates):
        cates[i] = str(i)
    val_iter = DataLoader(val_set, batch_size = 256, num_workers=16)

    confusion = ConfusionMatrix(num_classes=len(cates), labels=cates)
    total_1 = 0
    total_5 = 0
    max_val_len = 0
    iters = 0
    
    for X, y in val_iter:
        X, y = X.cuda(0), y.cuda(0)
        x1, x2, x3, x4, x5, out = model(X)
        ret, predictions = torch.max(out.data, 1)
        confusion.update(predictions.cpu().numpy(), y.cpu().numpy())
        # correct = (predictions == labels).sum()
        top1 = calculate_top_k_accuracy(out, y, 1)
        top5 = calculate_top_k_accuracy(out, y, 5)
        total_1 += top1*y.shape[0]
        total_5 += top5*y.shape[0]
        iters += 1
        max_val_len += y.shape[0]
        print('Iteration: {}/{}'.format(iters, len(val_iter)), 
                'top1: %.3f' % top1.float(), 
                'top5: %.3f' % top5.float())
    print('Top1 ACC: %.3f'%(total_1.float()/max_val_len), 
          'Top5 ACC: %.3f'%(total_5.float()/max_val_len))
    
    confusion.plot("./results/{}/{}_{}_{}_confusionmatrix.png".format(mode, netname, dataset, mode))
    _, avgp, avgp, avgs = confusion.summary()
    
    with open("./results/result.txt", "a+") as f:
        f.write("{}_{}_{}:\r\n".format(mode, netname, dataset))
        f.write("top1={}, top5={}, Precision={}, Recall={}, Specificity={}, params=none, gflops=none, size=none\r\n".format(
            total_1.float()/max_val_len, total_5.float()/max_val_len,
            avgp, avgp, avgs
        ))


    # return total_correct.float()/max_val_len

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
                ret, model = load_checkpoint("../weights/"+args.mode+"/"+args.net+"_"+dataset+"_best_singlegpu.pth")
                if not ret:
                    print("network {} not found".format(netname))
                    exit(1)
                model.cuda(args.gpus)
                with torch.no_grad():
                    model.eval()
                    test(cates, val_set, model, args.mode, dataset, netname)












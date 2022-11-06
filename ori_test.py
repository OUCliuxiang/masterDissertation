import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import prettytable as pt

from data import train_dataloader,train_datasets, val_datasets, val_dataloader

NUM_CLASSES = 48

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
        return str(acc)

    def plot(self):#绘制混淆矩阵
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
        plt.title('Confusion matrix (acc='+self.summary()+')')

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
        plt.savefig("confusionMatrix.png", dpi='figure', pad_inches=0.1)


def test():
    cates = [None] * NUM_CLASSES

    with open("../data/val.txt", 'r') as f:
        for line in  f.readlines():
            line = line.strip().split()
            line[0] = line[0].split('/')
            if cates[int(line[1])]:
                continue
            cates[int(line[1])] = line[0][-2]
    print(len(cates))
    print(cates)

    confusion = ConfusionMatrix(num_classes=len(cates), labels=cates)
    model.eval()
    total_1 = 0
    total_5 = 0
    val_iter = iter(train_dataloader)
    max_iter = len(train_dataloader)
    max_val_len = 0
    for iteration in range(max_iter):
        try:
            images, labels = next(val_iter)
        except:
            continue
        if torch.cuda.is_available():
            this_val_len = len(images)
            max_val_len += this_val_len
            images, labels = images.cuda(), labels.cuda()
            _, out = model(images)
            ret, predictions = torch.max(out.data, 1)
            confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
            # correct = (predictions == labels).sum()
            top1 = calculate_top_k_accuracy(out, labels, 1)
            top5 = calculate_top_k_accuracy(out, labels, 5)
            total_1 += top1*this_val_len
            total_5 += top5*this_val_len
            print('Iteration: {}/{}'.format(iteration, max_iter), 
                    'top1: %.3f' % top1.float(), 
                    'top5: %.3f' % top5.float())
    print('Top1 ACC: %.3f'%(total_1.float()/max_val_len), 
          'Top5 ACC: %.3f'%(total_5.float()/max_val_len))
    
    confusion.plot()
    confusion.summary()

    # return total_correct.float()/max_val_len

def calculate_top_k_accuracy(logits, targets, k=5):
    values, indices = torch.topk(logits, k=k, sorted=True)
    y = torch.reshape(targets, [-1, 1])
    correct = (y == indices) * 1.  # 对比预测的K个值中是否包含有正确标签中的结果
    top_k_accuracy = torch.mean(correct) * k  # 计算最后的准确率
    return top_k_accuracy


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model

model = load_checkpoint('../weights/mobilenetv2/mobilenetv2_1010_4599.pth')
# model = nn.DataParallel(model,device_ids=list(range(2)))

if torch.cuda.is_available():
    model.cuda()

test()
        
        











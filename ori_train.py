# -*- coding:utf-8 -*-

import torch
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from data import train_dataloader,train_datasets, val_datasets, val_dataloader
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step


##创建训练模型参数保存的文件夹
save_folder = cfg.SAVE_FOLDER + cfg.model_name
os.makedirs(save_folder, exist_ok=True)



def test():
    model.eval()
    total_correct = 0
    val_iter = iter(val_dataloader)
    max_iter = len(val_dataloader)
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
            prediction = torch.max(out, 1)[1]
            correct = (prediction == labels).sum()
            total_correct += correct
            print('Iteration: {}/{}'.format(iteration, max_iter), 'bs: {}'.format(this_val_len), 'ACC: %.3f' %(correct.float()/this_val_len))
    print('All ACC: %.3f'%(float(total_correct)/max_val_len))
    return total_correct.float()/max_val_len


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model

#####build the network model
if not cfg.RESUME_EPOCH:
    print('****** Training {} ****** '.format(cfg.model_name))
    print('****** loading the Imagenet pretrained weights ****** ')
    if not cfg.model_name.startswith('efficientnet'):
        print("05")
        model = cfg.MODEL_NAMES[cfg.model_name](num_classes=cfg.NUM_CLASSES)
        print("06")
        for name, p in model.named_parameters():
            print(name)
            # if c >=700:
            #     break
            p.requires_grad = True
        ## 冻结前边一部分层不训练
        '''
        ct = 0
        for child in model.children():
            ct += 1
            # print(child)
            if ct < 7:
                print(child)
                for param in child.parameters():
                    param.requires_grad = False
        '''
    else:
        print("07")
        model = cfg.MODEL_NAMES[cfg.model_name](cfg.model_name,num_classes=cfg.NUM_CLASSES)
        print("08")
        c = 0
        
        for name, p in model.named_parameters():
            c += 1
            print(c)
            # if c >=700:
            #     break
            p.requires_grad = True
        
if cfg.RESUME_EPOCH:
    print(' ******* Resume training from {}  epoch {} *********'.format(cfg.model_name, cfg.RESUME_EPOCH))
    model = load_checkpoint(os.path.join(save_folder, 'mobilenetv2_1010_4599.pth'))



##进行多gpu的并行计算
if cfg.GPUS>1:
    print('****** using multiple gpus to training ********')
    model = nn.DataParallel(model,device_ids=list(range(cfg.GPUS)))
else:
    print('****** using single gpu to training ********')
print("...... Initialize the network done!!! .......")

###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()


##定义优化器与损失函数
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
# optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                      momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()


lr = cfg.LR

batch_size = cfg.BATCH_SIZE

#每一个epoch含有多少个batch
max_batch = len(train_datasets)//batch_size
epoch_size = len(train_datasets) // batch_size
## 训练max_epoch个epoch
max_iter = cfg.MAX_EPOCH * epoch_size

start_iter = cfg.RESUME_EPOCH * epoch_size

epoch = cfg.RESUME_EPOCH

# cosine学习率调整
warmup_epoch=5
warmup_steps = warmup_epoch * epoch_size
global_step = 0

# step 学习率调整参数
stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
step_index = 0

max_ac = .0
for iteration in range(start_iter, max_iter):

    global_step += 1

    ##更新迭代器
    if iteration % epoch_size == 0:
        # create batch iterator
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1
        if epoch > 1:
            pass
        this_ac = test()
        ###保存模型
        model.train()
        if this_ac > max_ac:
            max_ac = this_ac
            if cfg.GPUS > 1:
                checkpoint = {'model': model.module,
                            'model_state_dict': model.module.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, '%.4f.pth' % max_ac))
            else:
                checkpoint = {'model': model,
                            'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, '%.4f.pth' % max_ac))

    if iteration in stepvalues:
        step_index += 1
    # lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)

    ## 调整学习率
    lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
                              learning_rate_base=cfg.LR,
                              total_steps=max_iter,
                              warmup_steps=warmup_steps)


    ## 获取image 和 label
    try:
        images, labels = next(batch_iterator)
    except:
        continue

    ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    _, out = model(images)
    loss = criterion(out, labels)
    # loss.requires_grad_(True)

    optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
    loss.backward()  # loss反向传播
    optimizer.step()  ##梯度更新

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    ##这里得到的train_correct是一个longtensor型，需要转换为float
    # print(train_correct.type())
    train_acc = (train_correct.float()) / batch_size

    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))













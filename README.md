0 号卡没有显存时需要使用CUDA_VISIBLE_DEVICES 指定显卡，否则这sb pytorch 就认定了在0卡走一遍莫名其妙的显存，避不开。

使用CUDA_VISIBLE_DEVICES指定显卡，意味着程序只能看到你指定的卡，比如指定 3,5 号卡，那程序只能看到这两个卡，也就是在程序内部，物理机3卡序号为0，物理机5号序号1

目前除了 baseline ，其他所有程序 net/dataset 只能用缺省 All 参数，批量训练，因为指定网络和数据集的逻辑还没来得及写 

参数传入的学习率是 sgd 优化器用的，一般用缺省的0.01就行，因为有写了衰减策略。adam 优化器传入固定的 3e-4

参数传入的 epoch 是最大 epoch，因为训练有提前终止策略。


这个准确率和过拟合问题，很迷，同一个程序，实验是在至少四台不同机器上跑的，环境各不相同。有时候很好，有时候很差,似乎和机器也没啥关系，这个没有仔细讨论过。可以多试几个随机数种子，比如设定种子 42，或者多试几个pytorch 版本，或者处理一下数据集，做一做数据清洗或者增广。多试几次，或者自己想想办法，是可以复现论文数据的。

指标：
1, weight size + compression ratio
2, weight size after Han + compression ratio
3, params + compression ratio
4, FLOPs + compression ratio
5, (top1 & top5) acc
6, (epochs & time consumption) to converge

作图：
1, loss & acc
2, acc with T or alpha
3, feature heat map 
4, confusion matrix
5, cate cluster
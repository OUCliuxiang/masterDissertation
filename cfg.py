import os

# 这里需要手动指定一个根目录
home = os.path.expanduser('~/wave/')
model_name = "mobilenetv2"

##数据集的类别
NUM_CLASSES = 48

#训练时batch的大小
BATCH_SIZE = 64

#网络默认输入图像的大小
INPUT_SIZE = 224
#训练最多的epoch
MAX_EPOCH = 500
# 使用gpu的数目
GPUS = 2
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3

BASE = home + 'data/'
# 训练好模型的保存位置
SAVE_FOLDER = home + 'weights/'
#数据集的存放位置
TRAIN_LABEL_DIR =BASE + 'train.txt'     
VAL_LABEL_DIR = BASE + 'val.txt'
TEST_LABEL_DIR = BASE + 'test.txt'






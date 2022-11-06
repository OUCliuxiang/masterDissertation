import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
# import cv2 as cv

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img

class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img


def get_train_transform(mean=mean, std=std, size=0):
    if size == 32 or size == 64:
        ratio = 1
    else:
        ratio = 256.0/224.0
    train_transform = transforms.Compose([
        Resize((int(size * ratio), int(size * ratio))),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(0.3)
        RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform

def get_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class MyDataSet(Dataset):
    def __init__(self, imageset, path, size):
        self.size = size
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*/*'))
        self.all_image_paths = [str(path) for path in all_image_paths]
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((label, index) for index, label in enumerate(label_names))
        self.all_image_labels = [label_to_index[path.parent.name] for path in all_image_paths]
        self.img_aug = True
        if imageset == "train":
            self.transform = get_train_transform(size = self.size)
        else:
            self.transform = get_test_transform(size = self.size)


    def __getitem__(self, index):
        img = Image.open(self.all_image_paths[index]).convert("RGB")
        if self.img_aug:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img))
        label = torch.tensor(self.all_image_labels[index])
        return img, label

    def __len__(self):
        return len(self.all_image_paths)


def get_dataset(path):
    if not path in ["caltech-101", "caltech-256", "cifar-10", "cifar-100", 
                "flower-102", "imagenet-mini", "svhn", "tiny-imagenet"]:
        print("path not found")
        return False, None, None
    elif "cifar" in path or "svhn" in path:
        return True, MyDataSet("train", "../data/"+path+"/train", 32), MyDataSet("test", "../data/"+path+"/val", 32)
    elif "tiny-imagenet" in path:
        return True, MyDataSet("train", "../data/"+path+"/train", 64), MyDataSet("test", "../data/"+path+"/val", 64)
    else:
        return True, MyDataSet("train", "../data/"+path+"/train", 224), MyDataSet("test", "../data/"+path+"/val", 224)


def get_net(dataset, net):
    if not net in ["vgg-t", "vgg-s", "mobilenet-t", "mobilenet-s", "resnet-t", "resnet-s"]:
        return False, None

    size = 0
    cates = 0
    if dataset == "caltech-101":
        size = 224
        cates = 102
    elif dataset == "caltech-256":
        size = 224
        cates = 257
    elif dataset == "cifar-10":
        size = 32
        cates = 10
    elif dataset == "cifar-100":
        size = 32
        cates = 100
    elif dataset == "flower-102":
        size = 224
        cates = 102
    elif dataset == "imagenet-mini":
        size = 224
        cates = 1000
    elif dataset == "svhn":
        size = 32
        cates = 10
    elif dataset == "tiny-imagenet":
        size = 64
        cates = 200
    else:
        print("wrong dataset path: {}".format(dataset))
        exit(1)
    
    if net == "vgg-t":
        from models.vgg import get_vgg
        return True, get_vgg('t', size, cates)
    elif net == "vgg-s":
        from models.vgg import get_vgg
        return True, get_vgg('s', size, cates)
    elif net == "mobilenet-t":
        from models.mobilenetv2 import get_mobilenetv2
        return True, get_mobilenetv2('t', size, cates)
    elif net == "mobilenet-s":
        from models.mobilenetv2 import get_mobilenetv2
        return True, get_mobilenetv2('s', size, cates)
    elif net == "resnet-t":
        from models.resnet import get_resnet
        return True, get_resnet('t', size, cates)
    elif net == "resnet-s":
        from models.resnet import get_resnet
        return True, get_resnet('s', size, cates)
    else:
        return False, None



def adjust_learning_rate_cosine(optimizer, global_step, learning_rate_base, total_steps, warmup_steps):

    lr = cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=warmup_steps,
                             hold_base_rate_steps=0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr



def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    # if global_step % (2*warmup_steps)==0:
    #     learning_rate = learning_rate_base * 0.1
    learning_rate = 0.3 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



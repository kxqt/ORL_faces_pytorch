import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps 

# 高通滤波
class HighPassFilter(object):
    def __init__(self, ext=8, shape=(100, 100)):
        self.ext = ext
        h, w = shape
        self.A = np.mat([[np.cos((j + 0.5) * np.pi * i / h) for j in range(w)] for i in range(h)])
        self.A = np.sqrt(2 / h) * self.A
        self.A[0, :] /= np.sqrt(2)
        
    def __call__(self, img):
        _img = img.numpy()
        fre = self.dct(_img[0])
        fre[:self.ext, :self.ext] = 0
        _img[0] = self.idct(fre)
        return torch.from_numpy(_img)
    
    def dct(self, img, flag=1):
        if flag == 1:
            res = self.A * np.mat(img) * self.A.T
        else:
            res = self.A.T * np.mat(img) * self.A
        return np.array(res)
    
    def idct(self, img):
        return self.dct(img, -1)

class leNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # 1 * 100 * 100
            nn.Conv2d(1, 16, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(6),
            nn.MaxPool2d((2,2)),
            # 6 * 48 * 48
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),
            # 16 * 22 * 22
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            # 32 * 10 * 10
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 20)#,
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 20)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class mcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            # 1 * 100 * 100
            nn.Conv2d(1, 16, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # 16 * 47 * 47
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # 32 * 22 * 22
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
            # 16 * 10 * 10
        )
        self.cnn2 = nn.Sequential(
            # 1 * 100 * 100
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # 20 * 48 * 48
            nn.Conv2d(20, 40, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # 40 * 22 * 22
            nn.Conv2d(40, 20, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
            # 20 * 10 * 10
        )
        self.cnn3 = nn.Sequential(
            # 1 * 100 * 100
            nn.Conv2d(1, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            # 12 * 98 * 98
            nn.Conv2d(12, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # 20 * 48 * 48
            nn.Conv2d(24, 48, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # 48 * 23 * 23
            nn.Conv2d(48, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
            # 24 * 10 * 10
        )
        self.cnn4 = nn.Sequential(
            # 60 * 10 * 10
            nn.Conv2d(60, 8, kernel_size=1),
            nn.ReLU(inplace=True)
            # 8 * 10 * 10
        )
        self.fc = nn.Sequential(
            nn.Linear(800, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16)
        )
        
    def forward_once(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        xx = torch.cat((x1, x2, x3), 1)
        xx = self.cnn4(xx)
        xx = xx.view(xx.size()[0], -1)
        xx = self.fc(xx)
        return xx

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

#定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=16):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            # 1 * 100 * 100
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2))
            # 64 * 50 * 50
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=2)
        # 64 * 25 * 25
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        # 128 * 12 * 12
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        # 256 * 6 * 6
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # 512 * 3 * 3
        self.fc = nn.Linear(512, num_classes)
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward_once(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

#自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1)
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True, pos_rate=0.5):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.positive_rate = pos_rate
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs) #37个类别中任选一个
        should_get_same_class = random.random() #保证同类样本约占一半
        if should_get_same_class < self.positive_rate:
            while True:
                #直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    

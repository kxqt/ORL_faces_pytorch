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

from models import *

device = 1

# 测试函数
def validate(test_loader, net, threshod=1.0):
    data_buf = []
    TP, FP, TN, FN = 0,0,0,0
    for it in test_loader:
        data_buf.append(it)
    for i, it1 in enumerate(data_buf):
        if i % 10 == 0:
            FAR = FP / (FP + TN) if FP + TN != 0 else -1
            FRR = FN / (FN + TP) if FN + TP != 0 else -1
            print(f"round {i} -- FAR: {FAR}, FRR: {FRR}")
        for j, it2 in enumerate(data_buf):
            if i != j:
                y = (it1[1] == it2[1])
                x0, x1 = it1[0], it2[0]
                output1,output2 = net(x0.cuda(device),x1.cuda(device))
                euclidean_distance = F.pairwise_distance(output1, output2)
                pred = (euclidean_distance < threshod)
                if y:
                    if pred:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if pred:
                        FP += 1
                    else:
                        TN += 1
    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)
    print(f"total: {FP + TN + FN + TP}")
    print(f"FAR: {FAR}, FRR: {FRR}, correct rate: {(TP + TN) / (FP + TN + FN + TP)}") 

testing_dir = "./data/test/"  #测试集地址
transform_test = transforms.Compose([transforms.Resize((100,100)), #有坑，传入int和tuple有区别
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize((0.4515), (0.1978)),
                                transforms.GaussianBlur(3),
                                    HighPassFilter()])
dataset_test = torchvision.datasets.ImageFolder(testing_dir, transform=transform_test)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=1)

best_net = torch.load('./model/t.pt')
validate(test_loader, best_net, 0.8)

#定义测试的dataset和dataloader

# #定义文件dataset
# testing_dir = "./data/test/"  #测试集地址
# folder_dataset_test = torchvision.datasets.ImageFolder(root=testing_dir)

# #定义图像dataset
# transform_vis = transforms.Compose([transforms.Resize((100,100)), #有坑，传入int和tuple有区别
#                                 transforms.ToTensor()])
# trans = transforms.Compose([transforms.Normalize((0.4515), (0.1978)),
#                                 transforms.GaussianBlur(3)])
# siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                              transform=transform_vis,
#                                         should_invert=False)

# #定义图像dataloader
# test_dataloader = DataLoader(siamese_dataset_test,
#                             shuffle=True,
#                             batch_size=1)


# #生成对比图像
# dataiter = iter(test_dataloader)
# # x0,_,_ = next(dataiter)

# for i in range(10):
#     x0,x1,label2 = next(dataiter)
#     concatenated = torch.cat((x0,x1),0)
#     _x0 = trans(x0)
#     _x1 = trans(x1)
#     output1,output2 = net(_x0.cuda(device),_x1.cuda(device))
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()),
#           path=f'./output/result_{i}.png')
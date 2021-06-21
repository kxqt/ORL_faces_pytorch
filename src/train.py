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

#定义一些超参
train_batch_size = 8        #训练时batch_size
train_number_epochs = 50     #训练的epoch

def imshow(img,text=None,should_save=False,path=None): 
    #展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy() #将tensor转为ndarray
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
    if path:
        plt.savefig(path)
    plt.show()    

def show_plot(iteration,loss,path=None):
    #绘制损失变化图
    plt.plot(iteration,loss)
    plt.ylabel("loss")
    plt.xlabel("batch")
    if path:
        plt.savefig(path)
    plt.show()
    
    
#定义文件dataset
training_dir = "./data/train/"  #训练集地址
folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

#定义图像dataset
transform = transforms.Compose([transforms.Resize((100,100)), #有坑，传入int和tuple有区别
                                transforms.ToTensor(),
                                transforms.Normalize((0.4515), (0.1978)),
                                transforms.GaussianBlur(3),
                                HighPassFilter()]) 
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False, pos_rate=0.5)

#定义图像dataloader
train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=train_batch_size)

#自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive

device = 1

# net = SiameseNetwork().cuda(device) #定义模型且移至GPU
net = leNet().cuda(device)
# net = mcnn().cuda(device)
# net = ResNet(ResBlock).cuda(device)
criterion = ContrastiveLoss() #定义损失函数
optimizer = optim.Adam(net.parameters(), lr = 0.0005) #定义优化器

counter = []
loss_history = [] 
iteration_number = 0


#开始训练
for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1 , label = data
        #img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1 , label = img0.cuda(device), img1.cuda(device), label.cuda(device) #数据移至GPU
        optimizer.zero_grad()
        output1,output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0 :
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch,loss_contrastive.item()))
    
show_plot(counter, loss_history, 'output/loss.jpg')

model_path = './model/t.pt'
torch.save(net, model_path)
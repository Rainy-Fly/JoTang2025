import numpy as np
import torch
from torch.nn.functional import conv2d
from torch.nn import MaxPool2d 
from torch import relu 
from torch.nn import CrossEntropyLoss 

punish=0.005
eps=5e-5
lossFunction=CrossEntropyLoss()

class Conv:
    convlist=[]
    Convlist=[]
    def __init__(self,size,cin,cout):
        self.c=torch.randn(cout,cin,size,size,requires_grad=True)
        self.c=torch.nn.init.kaiming_normal_(self.c,mode="fan_in",nonlinearity="relu")
        Conv.Convlist.append(self)
        Conv.convlist.append(self.c)
    def conving(self,image,pad=1):
        return conv2d(input=image,weight=self.c,stride=1,bias=None,padding=pad)


#最大池化
def max_pool(pre,stride):
    pool=MaxPool2d(kernel_size=stride,stride=stride)
    return pool(pre)

#自适应池
def adapt_ave_pool(pre):
    adapted=pre.mean(dim=[2,3])
    line=adapted
    return line

#标准化
def norm(imgin):
    mean_=imgin.mean(dim=[0,2,3],keepdim=True)
    var_=imgin.var(dim=[0,2,3],keepdim=True)
    imgout=(imgin-mean_)/(var_.sqrt()+eps)
    return imgout
 
#特征提取层
def In_Out(sample,convs,intermediate_outputs=False):
    conv0=convs[0]
    conv1=convs[1]
    conv2=convs[2]
    conv3=convs[3]
    conv4=convs[4]

    conved1=relu(norm(conv0.conving(sample,3)))
    conved2=relu(norm(conv1.conving(conved1)))
    pooled1=max_pool(conved2,2)
    conved3=relu(norm(conv2.conving(pooled1)))
    pooled2=max_pool(conved3,2)
    conved4=relu(norm(conv3.conving(pooled2)))
    pooled3=max_pool(conved4,2)
    conved5=relu(norm(conv4.conving(pooled3)))
    if intermediate_outputs==True :
        return (conved1,conved5)
    else:
        return adapt_ave_pool(conved5)


#全连接层
def linear(terminal,w,b):
    return terminal@w+b

def Back(y_hat,label,backlist,adam):
    loss=lossFunction(y_hat,label)
    l2_reg=0
    for reg in backlist:
        l2_reg+=(reg**2).mean()
    loss_=loss+l2_reg*punish
    loss_.backward()
    adam.step()
    adam.zero_grad()



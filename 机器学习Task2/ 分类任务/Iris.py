import torch 
import numpy
import random
import pandas as pd
from sklearn.datasets import load_iris
#数据预处理
batch=20
epoch=100
idata=load_iris()
eye=numpy.eye(3)
iX=numpy.float32(idata.data)
iy=idata.target
iy=eye[iy]
ptrs=(torch.randperm(len(iX))).numpy()
train=ptrs[:int(len(ptrs)*0.8)]
test=ptrs[int(len(ptrs)*0.8):]

def t(m):
    return torch.tensor(m)

W0=torch.randn(4,16,requires_grad=True)
W1=torch.randn(16,8,requires_grad=True)
W2=torch.randn(8,3,requires_grad=True)
b0=torch.randn(1,16,requires_grad=True)
b1=torch.randn(1,8,requires_grad=True)
b2=torch.randn(1,3,requires_grad=True)
ww=[W0,W1,W2]
bb=[b0,b1,b2]
ada=torch.optim.Adam(ww+bb,lr=0.02,betas=(0.9,0.999))
eps=5e-10

#前向传播函数
def calc(x):
    pout=x
    for i in range(3):
        pin=pout
        W=ww[i]
        b=bb[i]
        if i==2:
            return torch.softmax(pin@W+b,dim=1)
        pout=torch.tanh(pin@W+b)

#训练
for k in range(epoch):
    aBatch=random.sample(list(train),batch)
    X=t(iX[aBatch])
    y=t(iy[aBatch])
    y_hat=calc(X)
    loss=-(y*torch.log(y_hat+eps)).mean()
    loss.backward()
    ada.step()
    ada.zero_grad()  

#测试   
calced_y=calc(t(iX[test]))
test_y=iy[test]
same=0
matrix=numpy.zeros((3,3))
for i in range(test_y.shape[0]):
    v=torch.argmax(torch.tensor(test_y)[i])
    h=torch.argmax(calced_y[i])
    matrix[v][h]+=1
    if v==h:
        same+=1
acc=float(same/len(test))
df=pd.DataFrame(matrix,columns=["山鸢尾(setosa)","变色鸢尾(versicolor)","维吉尼亚鸢尾(virginica)"],index=["山鸢尾(setosa)","变色鸢尾(versicolor)","维吉尼亚鸢尾(virginica)"])
print(matrix)
print("---------------\t")
print("准确率为",acc)
print("---------------\t")
print(df)
print("---------------\t")
print("欢迎使用iris模型")
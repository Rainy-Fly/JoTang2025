import torch
import numpy
import matplotlib.pyplot as plt
import random
import time
from sklearn.datasets import make_moons

#超参数
sample=1000
lr=0.06
punish=5e-4
batch=40
epoch=2000
noise_=0.05
levels=2

#样本预处理
index=torch.randperm(sample)
train=index[:int(sample*0.8)]
test=index[int(sample*0.8):]
preX,prey=make_moons(n_samples=sample,noise=noise_)
preX=torch.tensor(preX,dtype=torch.float)
prey=torch.tensor(prey,dtype=torch.float).unsqueeze(1)


#权重与偏置
w0=torch.randn(2,16,requires_grad=True)
w1=torch.randn(16,8,requires_grad=True)
w2=torch.randn(8,1,requires_grad=True)

Ws=[w0,w1,w2]
b0=torch.randn(16,requires_grad=True)
b1=torch.randn(8,requires_grad=True)
b2=torch.randn(1,requires_grad=True)

bs=[b0,b1,b2]
last=False
eps=1e-7
losses={}
#前向传播函数
def neuro(X,Y):
    pout=X
    for level in range(0,levels+1):
        pin=pout.float()
        W=Ws[level]
        b=bs[level]
        if(level<levels):
            pout=torch.tanh(pin@W+b)
        else :
            pout=torch.sigmoid(pin@W+b)
    y_hat=pout
    if last==True :
        return y_hat
    else:
        y=Y
        loss=-(y*torch.log(y_hat+eps)+(1-y)*torch.log(1-y_hat+eps)).mean()+(sum(w.abs().sum() for w in Ws)/batch)*punish
        return loss

#训练开始
begin=time.time()
for i in range(epoch):
    fetch=random.sample(list(train),k=batch)
    y=prey[fetch,:]
    loss=neuro(preX[fetch,:],y)
    loss.backward() 
    losses[i]=float(loss)    
    with torch.no_grad():
        for k in Ws + bs: 
            k-=k.grad*lr
            k.grad.zero_()    

end=time.time()

#可视化处理
testx=preX[test]
testy=prey[test]
xmin,xmax,ymin,ymax=preX[:,0].min(),preX[:,0].max(),preX[:,1].min(),preX[:,1].max()
x1d,y1d=numpy.linspace(xmin,xmax,100),numpy.linspace(ymin,ymax,100)
x2d,y2d=numpy.meshgrid(x1d,y1d) 
preZ=(torch.stack((torch.tensor(x2d),torch.tensor(y2d)),dim=2))
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(7,8))

ax1.set_title("heatmap//热力图")
with torch.no_grad():
    last=True
    Z=neuro(preZ.view(10000,2),prey)
Z=Z.view(x2d.shape)
ax1.imshow(Z,cmap='RdYlBu_r',origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=0,vmax=1,aspect='auto')
ax1.contour(x2d,y2d,Z,levels=[0.5],colors='black',vmin=0,vmax=1,linestyle='--',linewidth='0.8')

ax2.set_title("the original moon//")
f=numpy.where(prey,'red','blue').ravel()
ax2.scatter(preX[:,0],preX[:,1],c=f,s=5)
ax2.set_aspect('auto')

ax3.set_title("loss function")
ax3.plot(list(losses.keys()),list(losses.values()),linewidth=0.2)


fig.suptitle("make_moons可视化")

#plt.show()
print("耗时",end-begin,"s")
      


  


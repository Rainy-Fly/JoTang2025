import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim

#标准化和反标准化
class NormOne:
    def __init__(self,y):
        self.Mean=y.mean()
        self.std=y.std()
        self.y_=(y-self.Mean)/self.std
    def norm(self):
        return self.y_
    def backnorm(self,y_):
        return y_*self.std+self.Mean
    
#超参数
lr=0.0001
punish=0.0001
batch=40
epoch=100
housing=pd.read_csv("被加载的csv文件/cal_housing.csv",header=None)
sample=torch.tensor(housing.iloc[:,:-1].to_numpy(), dtype=torch.float32)
for i in range(sample.size(1)):
    column=sample[:,i]
    Mean=column.mean()
    std=column.std()
    sample[:,i]=(column-Mean)/std
Norm_one=NormOne(torch.tensor(housing.iloc[:,-1].to_numpy(), dtype=torch.float32))
targets=Norm_one.norm()

RandomIndex=list(torch.randperm(len(sample)).numpy())
trainIndex=RandomIndex[:int(len(RandomIndex)*0.8)]
testIndex=RandomIndex[int(len(RandomIndex)*0.8):int(len(RandomIndex)*0.9)]
valIndex=RandomIndex[int(len(RandomIndex)*0.9):]
train_x=sample[trainIndex]
train_y=targets[trainIndex]
val_x=sample[valIndex]
val_y=targets[valIndex]
test_x=sample[testIndex]
test_y=targets[testIndex]

#权重与偏置
lv1,lv2,lv3,lv4=64,128,32,8
w0=torch.randn(8,lv1,requires_grad=True)
w1=torch.randn(lv1,lv2,requires_grad=True)
w2=torch.randn(lv2,lv3,requires_grad=True)
w3=torch.randn(lv3,lv4,requires_grad=True)
w4=torch.randn(lv4,1,requires_grad=True)
Ws=[w0,w1,w2,w3,w4]
b0=torch.randn(lv1,requires_grad=True)
b1=torch.randn(lv2,requires_grad=True)
b2=torch.randn(lv3,requires_grad=True)
b3=torch.randn(lv4,requires_grad=True)
b4=torch.randn(1,requires_grad=True)
bs=[b0,b1,b2,b3,b4]
adam=optim.Adam(Ws+bs,lr)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(adam,mode="max",factor=0.5,patience=5)
with torch.no_grad():
    for w in Ws:
        torch.nn.init.kaiming_normal_(w,mode="fan_in",nonlinearity="leaky_relu")   
    for b in bs:    
        torch.nn.init.zeros_(b)

eps=5e-5

def See(x):
    print(x.size())

#前向传播函数
def neuro(X,Y,test=False):
    pout = X
    Y = Y.float().reshape(-1,1)
    for k in range(len(Ws)):
        pin=pout.float()
        W=Ws[k]
        b=bs[k]
        if k<len(Ws)-1:
            pout=torch.nn.functional.leaky_relu(pin@W+b)
        else :
            pout=(pin@W+b)
    loss = ((pout - Y) ** 2).mean()
    L2=sum((w**2).sum() for w in Ws)*punish
    if test==True:
        y_mean = Y.mean()
        R = 1 - (torch.sum((pout - y_mean) ** 2) / (torch.sum((Y - y_mean) ** 2) + 1e-12))
        print(R.item())
        return (pout,loss,R)
    return loss+L2

def shuffle(x,y):
    xy=torch.concat((x,y.unsqueeze(1)),dim=1)
    xy=xy[torch.randperm(xy.size(0))]
    x=xy[:,:-1]
    y=xy[:,-1]
    return (x,y)

#训练
aveRs=[]
for j in range(epoch):
    Rs=[]
    train_x,train_y=shuffle(train_x,train_y)
    val_x,val_y=shuffle(val_x,val_y)
    for i in range(int((len(train_x)/batch))):
        batch_train=range(i*batch,(i+1)*batch)
        y=train_y[batch_train]
        loss_=neuro(train_x[batch_train],y)
        loss_.backward()
        with torch.no_grad():
            adam.step()   
            adam.zero_grad()
    for k in range(int((len(val_x)/batch))):
        batch_val=range(k*batch,(k+1)*batch)
        _,_,R=neuro(val_x[batch_val],val_y[batch_val],test=True) 
        Rs.append(R)
    scheduler.step(sum(Rs)/len(Rs))
    aveR=sum(Rs)/len(Rs)
    aveRs.append(aveR.detach().numpy())


#验证
predict,MSE,R=neuro(test_x,test_y,True)
predict_=Norm_one.backnorm(predict)
test_y_=Norm_one.backnorm(test_y)
print("MSE=",MSE.item(),"R=",R.item())

_,(ax1,ax2)=plt.subplots(1,2,figsize=(8,8))
ax1.scatter(predict_.detach().numpy(),test_y_.detach().numpy())
ax1.set_xlabel("predicted")
ax1.set_ylabel("real")
ax1.set_title("predict")
ax2.plot(aveRs)
ax2.set_title("R")
plt.show()

  


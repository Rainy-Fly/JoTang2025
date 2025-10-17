import pandas as pd 
import torch
import random
import matplotlib.pylab as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau as Reduce
from torch.nn.utils import clip_grad_norm_ as Clip


epoch=2000
lr_=0.00013
batch=40
punish=0.00009


betas_=(0.9,0.999)
eps=5e-5
lv1=128#第一层神经元数量
lv2=64#第二层神经元数量
func=torch.nn.LeakyReLU(0.1)



#原数据清洗
df=pd.read_csv("被加载的csv文件/titanic.csv")
Mode=["Age","Fare","Embarked"]
Other=["Survived","Pclass","Sex","SibSp","Parch"]
filledDf=pd.concat([df[Mode].fillna(df[Mode].mode().iloc[0]),df[Other]],axis=1)
encodedDf=pd.get_dummies(filledDf,columns=["Embarked","Sex"]).astype(int)
liveDF=encodedDf.query("Survived==1")
dieDF=encodedDf.query("Survived==0")
ratio=len(liveDF)/(len(liveDF)+len(dieDF))
live_weight=(1-ratio)/ratio

#划分训练集、验证集、测试集
trainDF=pd.concat([liveDF.sample(frac=0.8,random_state=1),dieDF.sample(frac=0.8,random_state=1)]).sample(frac=1)
validDF=pd.concat([liveDF.sample(frac=0.1,random_state=1),dieDF.sample(frac=0.1,random_state=1)]).sample(frac=1)
testDF=pd.concat([liveDF.sample(frac=0.1,random_state=1),dieDF.sample(frac=0.1,random_state=1)]).sample(frac=1)

train=torch.tensor(trainDF.to_numpy(),dtype=torch.float32)
valid=torch.tensor(validDF.to_numpy(),dtype=torch.float32)
test=torch.tensor(testDF.to_numpy(),dtype=torch.float32)


 
def std(samplepool):
    mean_=samplepool.mean(0)
    std_ =samplepool.std(0).clamp_min(1e-8)
    return (samplepool-mean_)/std_
train=torch.concat([std(train[:,:-1]),train[:,-1].unsqueeze(1)],dim=1)
valid=torch.concat([std(valid[:,:-1]),valid[:,-1].unsqueeze(1)],dim=1)
test=torch.concat([std(test[:,:-1]),test[:,-1].unsqueeze(1)],dim=1)


#权重，偏置，优化器
w0=torch.randn(train.shape[1]-1,lv1,requires_grad=True,dtype=torch.float32)
w1=torch.randn((lv1,lv2),requires_grad=True,dtype=torch.float32)
w2=torch.randn((lv2,1),requires_grad=True,dtype=torch.float32)
Ws=[w0,w1,w2]
b0=torch.randn(lv1,requires_grad=True,dtype=torch.float32)
b1=torch.randn(lv2,requires_grad=True,dtype=torch.float32)
b2=torch.randn(1,requires_grad=True,dtype=torch.float32)
bs=[b0,b1,b2]
adam=torch.optim.Adam(Ws+bs,lr=lr_,betas=betas_)
f1Scheduler=Reduce(adam,"max",factor=0.5,patience=5,eps=eps)

#归一化
with torch.no_grad():
    for w in Ws:
        torch.nn.init.kaiming_normal_(w,mode="fan_in",nonlinearity="leaky_relu")   
    for b in bs:    
        torch.nn.init.zeros_(b)

#定义前向传播函数
def one_batch(x):
    pout=x
    for i in range(3):
        w=Ws[i]
        b=bs[i]
        pin=pout
        if i==2:
            return torch.sigmoid(pin@w+b)
        pout=func(pin@w+b)

#定义计算F1-score的函数
def F1(y0,y1,samplepool):
    TP=0
    TN=0
    FP=0
    FN=0
    boarder=0.6
    for i in range(int(len(samplepool)/batch)):
        for j in range(len(y0[i])):
            y1_=y1[i][j]
            y0_=y0[i][j]
            if y1_>boarder and y0_==1:TP+=1
            if y1_<boarder and y0_==1:FN+=1
            if y1_>boarder and y0_==0:FP+=1
            if y1_<boarder and y0_==0:TN+=1
    acc=(TP+TN)/(TP+FN+FP+TN)
    premise=TP/(FP+TP+eps)
    recall=TP/(FN+TP+eps)
    score=(2*premise*recall)/(premise+recall+eps)
    return (acc,premise,recall,score)

#一轮epoch
Yhat=[]
Y=[]
def one_epoch(samplePool):
    for j in range(int(len(samplePool)/batch)):
        index_=random.sample(range(len(samplePool)),batch)
        x=samplePool[index_,:-1]
        y=samplePool[index_,-1]
        y_hat=one_batch(x)
        y_hat=y_hat.clamp(min=eps,max=1-eps)
        weight=torch.where(y==1,live_weight,1.0)
        #print(y_hat.size(),weight.size())
        y_hat=(y_hat.squeeze(1)*weight)
        #print(y_hat.size())
        if (samplePool is train):
            loss=-(y*torch.log(y_hat+eps)+(1-y)*torch.log(1-y_hat)).mean()
            loss_=loss+punish*((torch.cat(list(i.flatten() for m in (Ws,bs) for i in m ),dim=0))**2).mean()
            loss_.backward()
            Clip(Ws+bs,max_norm=0.5)
            adam.step()
            adam.zero_grad()
        if (samplePool is valid )or (samplePool is test) :
            with torch.no_grad():
                Y.append(y)
                Yhat.append(y_hat)

#每轮epoch记录
past_acc=[]
past_loss=[]
past_premise=[]
past_recall=[]
past_score=[]
pastPool=[past_acc,past_premise,past_recall,past_score]
Params=[]
best_f1=0
best_acc=0
best_recall=0
best_premise=0
patience=10

def show_figures():
    print("acc=",ave(past_acc))
    print("premise=",ave(past_premise))
    print("recall=",ave(past_recall))
    print("score=",ave(past_score))

def ave(lis):
    return sum(lis)/len(lis)

def validation():
    global best_f1,patience,f1,Y,Yhat,best_acc,best_premise,best_recall
    one_epoch(valid)
    count=0
    for i in F1(Y,Yhat,valid):
        pastPool[count].append(i)
        count+=1
    f1=pastPool[-1][-1]
    best_acc=max(pastPool[0][-1],best_acc)
    best_recall=max(pastPool[-2][-1],best_recall)
    best_premise=max(pastPool[1][-1],best_premise)
    f1Scheduler.step(f1)  
    Y=[]
    Yhat=[] 

#正式训练
def train_valid():
    global best_f1,Params,lr_,patience,past_acc,past_loss,past_premise,past_recall,past_score,pastPool
    for i in range(epoch):
        one_epoch(train)     
        validation()
        if i<50:continue
        if i%10==0:   
            show_figures()
            print(i,"--------")
        if f1>best_f1:
            best_f1=f1
            patience=10
            Params.append([Ws,bs,past_score[-1]])  
        if f1<best_f1-0.1:
            patience-=1
            if patience<=0:
                break
    past_acc=[]
    past_loss=[]
    past_premise=[]
    past_recall=[]
    past_score=[]    
    pastPool=[past_acc,past_premise,past_recall,past_score]
    
#主循环
while(best_f1<=0.72 or best_acc<0.72 or best_premise<0.72 or best_recall<0.72):
    best_acc,best_f1,best_premise,best_recall=0,0,0,0
    train_valid()
    

#合并训练
train=torch.cat([train,valid],dim=0)
for i in range(train.size(0)):
    one_epoch(train)
    print("合并后第",i,"轮循环")

#测试
max_index=max(range(len(Params)),key=lambda i :Params[i][-1])
Ws,bs,_=Params[max_index]
one_epoch(test)
count=0
pastPool=[past_acc,past_premise,past_recall,past_score]
for i in F1(Y,Yhat,valid):
    pastPool[count].append(i)
    count+=1
print("最终结果:")
show_figures()



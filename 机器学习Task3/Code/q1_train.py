from model import Conv,In_Out,linear,Back
from data import img
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as Reduce
from sklearn.metrics import precision_recall_fscore_support as f1_
import time
import torch
#卷积核
conv0=Conv(7,3,16)
conv1=Conv(3,16,32)
conv2=Conv(3,32,48)
conv3=Conv(3,48,64)
conv4=Conv(3,64,80)
ConvsList=Conv.Convlist

#超参数
epoch=5
batch=40
lr=0.001
#调整工具
w=None
b=None
adam=None
used=False
scheduler=None


eps=5e-5
past=time.time()
backlist=[]
bestParams=[]
epoch_ScoreList=[]
epoch_PreciseList=[]
epoch_RecallList=[]
batch_ScoreList=[]
batch_PreciseList=[]
batch_RecallList=[]
#训练集
train_batch_count=0
def train_batch(images,label):
    global w,b,adam,used,scheduler,backlist,train_batch_count
    print("train_batch",train_batch_count,"激活")
    out=In_Out(images,ConvsList)
    if used==False:
        wlen=out.size(1)
        w=torch.randn(wlen,10,requires_grad=True)
        b=torch.randn(10,requires_grad=True)
        backlist=Conv.convlist+[w,b]
        adam=Adam(backlist,lr=lr,betas=[0.9,0.999])
        scheduler=Reduce(adam,mode="max",factor=0.5,patience=5,eps=eps)
        used=True
    lined=linear(out,w,b)
    Back(lined,label,backlist,adam)
    train_batch_count+=1

#验证集
val_batch_count=0
def val_batch(images,label):
    global batch_PreciseList,batch_RecallList,batch_ScoreList,val_batch_count
    print("val_batch",val_batch_count," 开始")
    out=In_Out(images,Conv.Convlist)
    lined=linear(out,w,b)
    recall,precise,score=F1(lined,label)
    batch_PreciseList.append(precise)
    batch_RecallList.append(recall)
    batch_ScoreList.append(score)
    val_batch_count+=1


def F1(Predict,Label):
    predict=torch.argmax(Predict,dim=1).numpy()
    label=Label.numpy()
    precise,recall,score,_=f1_(predict,label)
    return (ave(recall),ave(precise),ave(score))           


def ave(nums):
    return sum(nums)/len(nums)

bestScore=0
bestPrecise=0
bestRecall=0
patience=5

def evaluate(recall,precise,score):
    global patience
    if recall<bestRecall or precise<bestPrecise or score<bestScore:
        patience-=1
    if patience<=0:
        return True
    else:
        return False

print("Epoches激活")
PredictList=[]
LabelList=[]
for i in range(epoch):
    print(i,"epoch开始")
    #报错后注释掉：
    for train,trainLabel in img("train",batch):
        train_batch(train,trainLabel)
    train_batch_count=0
    torch.save((w,b,lr),"wblr3.pt")
    #报错后注释掉;

    #下面的用于保存训练集数据，防止后续报错又要重新训练，报错后启用:
    #w,b,lr=torch.load("wblr.pt")
    #backlist=Conv.convlist+[w,b]
    #adam=Adam(backlist,lr=lr,betas=[0.9,0.999])
    #scheduler=Reduce(adam,mode="max",factor=0.5,patience=5,eps=eps)
    #报错后使用;

    for val,valLabel in img("val",batch):
        val_batch(val,valLabel)
    val_batch_count=0
    score=ave(batch_ScoreList)
    precise=ave(batch_PreciseList)
    recall=ave(batch_RecallList)
    scheduler.step(score)
    bestScore=max(bestScore,score)
    bestPrecise=max(bestPrecise,precise)
    bestRecall=max(bestRecall,recall)
    epoch_ScoreList.append(score)
    epoch_PreciseList.append(precise)
    epoch_RecallList.append(recall)
    backlist=[]
    if i%10==0:
        print("第",i,"轮val:","recall=",recall,",precise=",precise,",score=",score)
    if evaluate(recall,precise,score):
        patience=5
        print("第",i,"轮 break ")
        break
    else:
        bestParams=ConvsList.copy()+[w,b]
    print(i,"epoch结束")
        
#从训练集分出来5000样本自己测试
convs=bestParams[:-2]
w=bestParams[-2]
b=bestParams[-1]
ttest_RecallList=[]
ttest_PreciseList=[]
ttest_ScoreList=[]
for ttest,ttestLabel in img("ttest",batch):
    terminal=In_Out(ttest,convs)
    predict=linear(terminal,w,b)
    recall,precise,score=F1(predict,ttestLabel)
    print("ttest:","recall=",recall,"precise=",precise,"score=",score)
    ttest_PreciseList.append(precise)
    ttest_RecallList.append(recall)
    ttest_ScoreList.append(score)
    
print("recall=",ave(ttest_RecallList),"precise=",ave(ttest_PreciseList),"score=",ave(ttest_ScoreList))
if input("保存模型吗,[y/n]")=="y":
    torch.save(bestParams,"q1_model.pt")

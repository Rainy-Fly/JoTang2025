from torchvision import transforms
from torchvision.datasets import ImageFolder  
from torch.utils.data  import DataLoader as DL
from pathlib import Path
from sklearn.model_selection import train_test_split as Split
from torch.utils.data import Subset
dur=Path(__file__).resolve().parent.parent
trainDir=dur/"photo_resourse/train"
valDir=dur/"photo_resourse/val"
testDir=dur/"photo_resourse/test"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

trainData=ImageFolder(trainDir,transform=transform)
valData=ImageFolder(valDir,transform=transform)
testData=ImageFolder(testDir,transform=transform)

trainNum=len(trainData)
valNum=len(valData)
testNum=len(testData)

TrainIndice=range(trainNum)
TrainLabels=trainData.targets

train_index,test_index,_,_=Split(TrainIndice,TrainLabels,train_size=0.88888)
trainData_=Subset(trainData,train_index)
ttestData_=Subset(trainData,test_index)

train_Num=len(trainData_)
ttest_Num=len(ttestData_)

def img(choice,batch):
    if choice=="train":
        return DL(trainData_,batch_size=batch,shuffle=True)
    if choice=="val":  
        return DL(valData,batch_size=batch,shuffle=True)
    if choice=="ttest":
        return DL(ttestData_,batch_size=batch,shuffle=True)
    if choice=="test": 
        return DL(testData,batch_size=batch,shuffle=False)
        




    

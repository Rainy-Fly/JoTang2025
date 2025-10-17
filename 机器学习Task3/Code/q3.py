import torch
import matplotlib.pyplot as plt
from data import img
from model import In_Out

batch=40
bestParams=torch.load("results/q1_model.pt")
convs=bestParams[:-2]
w=bestParams[-2]
b=bestParams[-1]
Conv1_L2=torch.zeros((10,16))
Conv5_L2=torch.zeros((10,80))
count=0
turns=0
for images,label in img("val",batch):
    if turns%10!=0:continue
    conved1,conved5=In_Out(images,convs,intermediate_outputs=True)
    sumed1=torch.sum(conved1**2,dim=(2,3))
    sumed5=torch.sum(conved5**2,dim=(2,3))
    for h in range(16):
        for v in range(10):
            Conv1_L2[label[v],h]+=sumed1[v,h]
            count+=1
    for h in range(80):
        for v in range(10):
            Conv5_L2[label[v],h]+=sumed5[v,h]
    print("count=",count)
    if count>=6800 :break
Conv1_L2=Conv1_L2/count
Conv1_L2=(Conv1_L2-Conv1_L2.min( ))/(Conv1_L2.max()-Conv1_L2.min())
Conv5_L2=Conv5_L2/count  
Conv5_L2=(Conv5_L2-Conv5_L2.min())/(Conv5_L2.max()/Conv5_L2.min()) 

def draw_bar(L2Tensor,colunmn,level):
    x_axis=range(10)
    y_axis=L2Tensor[:,colunmn].detach().numpy()
    plt.bar(x_axis,y_axis)
    plt.title(f"conv-{level}-{column}'s L2")
    plt.xlabel("category")
    plt.ylabel("L2")

column=0
while True:
    #column=input("查看第一层的第几个卷积核(<16)")
    if column>=15:
        break
    draw_bar(Conv1_L2,int(column),1)
    plt.savefig(f"L2/1-{column}.png")
    plt.close()
    column+=1

column=0
while True:
    #column=input("查看第五层的第几个卷积核(<80)")
    if column>=79:
        break
    draw_bar(Conv5_L2,int(column),5)
    plt.savefig(f"L2/5-{column}.png")
    plt.close()
    column+=1

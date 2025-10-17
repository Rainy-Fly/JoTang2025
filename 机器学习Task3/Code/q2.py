import torch
import matplotlib.pyplot as plt


bestParams=torch.load("q1_model.pt")
Conv=bestParams[0].c.detach()

def draw(Tensor3,n):
    n=n
    Tensor3=(Tensor3-Tensor3.min())/(Tensor3.max()-Tensor3.min())
    fig,(ax1,ax2,ax3)=plt.subplots(1,3)
    #print(Tensor3[0].size())
    ax1.imshow(Tensor3[0],cmap='coolwarm')
    ax2.imshow(Tensor3[1],cmap='coolwarm')
    ax3.imshow(Tensor3[2],cmap='coolwarm')
    plt.title(f"the{n}conv")
    plt.savefig(f"第一层卷积核可视化/{n}.png")
    plt.close()

for i in range(Conv.size(0)):
    conv=Conv[i,:,:,:]
    draw(conv,i+1)
    i+=1
    
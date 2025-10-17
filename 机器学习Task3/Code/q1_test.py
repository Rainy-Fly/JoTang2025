import torch
from data import img
from model import In_Out,linear
bestParams=torch.load("results/q1_model.pt")
convs=bestParams[:-2]
w=bestParams[-2]
b=bestParams[-1]
batch=40
count=0
with open("q1_result.txt", "a", encoding="utf-8") as f:
    for images ,label in img("test",batch):
        terminal=In_Out(images,convs)
        lined=linear(terminal,w,b)
        _,predict=lined.max(dim=1)
        for i in predict.numpy():
            f.write(f"image_{count}.png,{i}\n")
            count+=1
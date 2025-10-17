# 实验报告
## 一.代码整体思路
    框架:
        1.data.py用ImageFolder获取图片并分类,将DataLoader包装成函数img,方便训练,验证,测试抽取各自集的样本

        2.model.py:由于卷积网络中卷积核,池化,标准化重复出现,我将其分别用类和函数实现,并整合了整个特征提取层的函数和全连接层函数,供train.pyo调用

        3.train.py:有了model的函数,容易完成单个epoch的框架;但我在每个epoch训练后都用验证集val跑一遍当前模型,计算F1score,precise,recall,把score传给Reduce调度器,实现每epoch更新lr,调整Adam优化方式.由于val已经在多个epoch用,我自己测试时又不能用test,我从45000个teain样本中抽取了5000当作测试集,计算F1score,precise,recall,目前各项指标在0,7-0.8附近
## 二.结果分析
### (1)第一层卷积核的图像分析
#### 1.识别颜色的卷积核
![](/results/可视化/第一层卷积核可视化/10.png)
这张图片的Red通道为浅色,数值小,响应不敏感,Blue通道为亮色,数值大,响应敏感    
说明这个卷积核学习的是图片的Blue蓝色特征

#### 2.识别三角形的卷积核
![](/results/可视化/第一层卷积核可视化/2.png)
R通道明显可见右上三角高亮,G通道左下三角亮,说明n两个卷积核分别在两个通道学习不同方向的三角特征
#### 3.学习边缘轮廓的卷积核
![](/results/可视化/第一层卷积核可视化/8.png)
最后一个通道的左侧和中间都是浅色,而右边可见半椭圆形亮色区域,说明他在学习图片的左轮廓
### (2)第一和五层卷积核L2范数h柱状图分析
1.第一层卷积核的响应特征
第一层卷积核学习的是简单特征,在不同类别中都具有普遍性,因此差异不大且都能提取到比较多的低级特征(线条,明暗,轮廓),因此响应明显且平均
![第一层第11个卷积核](/results/可视化/L2/1-11.png)
2.第五层卷积核特征
第五层拿到的特征图经过了前面卷积层的特征提取,识别到的是高级特征,而高级特征具有特异性,每一类拥有独一无二的高级特征,如猫的耳朵,尾巴,汽车的流线型构造,轮船的宽底.因此,第五层的卷积核通常只对某一两类敏感,L2不均匀.
![](/results/可视化/L2/5-58.png)
![](/results/可视化/L2/5-15.png)

## 三.困难及解决方案

### 1.文件导入
操作ImageFolder时,以为相对路径```~/Codes/机械学习3/photo_sourse```i只是n省略了绝对路径的前部分,担心和绝对路径一样无法在其他电脑上正常运行,使用data.py的Path获得父目录dur,再拼接```trainDir=dur/"photo_resourse/train" valDir=dur/"photo_resourse/val" testDir=dur/"photo_resourse/test"```;不断调整路径,增删放图片的文件夹,知晓了ImageFolder读取图片时传给他的文件夹即是他读的最大单位,下面的子文件夹会被他对应为为标签.
### 2.卷积和池化操作
最初使用嵌套for循环,滑动n窗口并不断纠结(y,x)在原特征图与卷积/池化后的图以及padding,窗口大小之间的数学关系;因为坐标从0开始计数,距离值和坐标值从左加和从右减要+1 or-1or不操作,经过反复调整:
```python
    def conving(convs,image,pad,outchannel):
    c=0
    global batch
    L=int(image.size(2))
    r=int((convs[2].size(2)-1)/2)
    newL=L+2*pad
    image_=torch.zeros(batch,image.size(1),newL,newL)
    image_[:,:,pad:L+pad,pad:L+pad]=image
    out=torch.empty(batch,outchannel,newL-2*r,newL-2*r)
    for conv in convs:
        for channel in range(image.size(1)):
            for y in range(r,newL-r):
                for x in range(r,newL-r):
                    window=image_[:,:,y-r:y+r+1,x-r:x+r+1]
                    core=torch.mul(conv,window).sum()
                    out[:,c,y-r,x-r]+=core
        c+=1   
    return out
```

最终循环几次后内存耗尽,VScode无响应...
于是换成```from torch.nn import MaxPool2d 
        self.c=torch.nn.init.kaiming_normal_(self.c,mode="fan_in",nonlinearity="relu")```
~~愉快地~~体验到了由矩阵运算,性能之王C++,并行处理而来的高效性
### 3.张量的维度问题
torch.cat/stack,tensor.max/mean/std,flatten,view,reshape都涉及维度操作,每次遇到都令人头疼,最终我认为都可以从permute理解.以三维矩阵为例,原本的维度dim=[0,1,2],permuten将维度调换,很抽象,其实可以理解为将立方体转换视图,如(0,2,1)将原本的宽w和H轴调换.以此为基础,stack不传dim时默认dim=0,在升维的新轴方向放置tensor,若传了dim=a,则是在dim=0的基础上permute(a,..,0,..)调换a与0维;cat则是简单得在原tensor的n维度拼接;max,mean,std等计算可以传一个或多个dim,但都是每次遍历未被传入的维度a相同,而被传入维度不同的所有元素,对他们进i求和,最大值,方差等操作.
### 4.循环中储存指标的方法
验证集val的batch级循环中,我先把所有预测值储存在list or tensor(用上文的cat),VScode再次炸掉,更不必说epoch级的容器来储存了.因此我在每轮batch拿预测值和实际标签计算f1score,并储存一个标量在每轮epoch重新回收内存的集合,再在每轮epoch计算平均数(调用提前定义的ave函数)
### 5.不同文件之间的关系依赖的问题
(1).data.py中定义了批量抽取样本和标签的函数,不让后面的文件import DataLoader和trainData;但我最初的img函数只有一个形参choice,DL中的batch由从外部捕获,需要```from train.py```,因此我将btach设定为形参,调用时传入
```python
def img(choice,batch):
    #DL为DataLoader
    if choice=="train":
        return DL(trainData_,batch_size=batch,shuffle=True)
    if choice=="val":  
        return DL(valData,batch_size=batch,shuffle=True)
    if choice=="ttest":
        return DL(ttestData_,batch_size=batch,shuffle=True)
    if choice=="test": 
        return DL(testData,batch_size=batch,shuffle=False)
```
(2).最初我在model中实现batch级抽取和epoch级别训练,让train.py中只需调用Train和Val函数,但这样model文件极其复杂混乱,train文件代码量又太少.因此最后我只在model中实现抽象的前向特征提取和线性层工具函数,train中再单独对train val ttest给各自的batch和epoch;让后方文件不会被前面的文件引用.

    

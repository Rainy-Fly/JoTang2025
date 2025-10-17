import nltk,torch,gensim
import torch.nn as nn
import numpy as np
from collections import Counter

#超参数
vector_size=100
window=5
min_count=2
epochs=10
ngram=1          

#取语料
nltk.download('brown',quiet=True)
sents=[[w.lower() for w in s] for s in nltk.corpus.brown.sents()]
print("取到语料")

#构造词汇表
word_counts=Counter(w for s in sents for w in s)
filtered_words=[w for w,c in word_counts.items() if c>=min_count]
vocab={w:i for i, w in enumerate(filtered_words)}  #确保索引连续
idx2w={i:w for w, i in vocab.items()}

#skip-gram
def train_sg(sents,vocab,window,len(vocab),vector_size,epochs):
    W=nn.Embedding(len(vocab),vector_size)
    C=nn.Embedding(len(vocab),vector_size)
    opt=torch.optim.Adam(list(W.parameters())+list(C.parameters()),lr=0.01)

    pairs=[]                                          
    for sent in sents:
        idxs=[vocab[w] for w in sent if w in vocab]
        for i,wid in enumerate(idxs):
            ctx=idxs[max(0,i-window):i]+idxs[i+1:i+1+window]
            pairs.extend([(wid,cid) for cid in ctx])
    pairs=torch.tensor(pairs,dtype=torch.long)        
    for _ in range(epochs):
        #i=0  demo版本改bug时用的
        for b in torch.split(pairs,512):                  
            wi,ci=b[:,0], b[:,1]                     
            score=(W(wi)*C(ci)).sum(1)
            loss=-torch.log(torch.sigmoid(score)).mean()
            opt.zero_grad();loss.backward();opt.step()
            print("一个样本 loss:",loss.item())            
            #i+=1
            #if i>20:   
            #    break
    return W.weight.data.numpy()


emb=train_sg(sents,vocab,window,len(vocab),vector_size,epochs)
print("训练出词向量矩阵")
model=gensim.models.KeyedVectors(vector_size=vector_size)
model.add_vectors([idx2w[i] for i in range(len(vocab))],emb)
print('与government相似度最高的:',[w for w, _ in model.most_similar('government',topn=5)])
print('king-man+woman=:',[w for w, _ in model.most_similar(positive=['king','woman'],negative=['man'], topn=1)])
print('computer与machine的相似度:',round(model.similarity('computer','machine'),3))

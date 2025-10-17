import nltk, collections, itertools
nltk.data.path.insert(0,'./nltk_data')     
from nltk.corpus import brown

#语料
sents=brown.sents()
text=list(itertools.chain.from_iterable(sents))
text=[w.lower() for w in text]
PAD="<s>"
vocab=set(text)
vocab_size=len(vocab)

#统计词频
unigram=collections.Counter(text)
bigram=collections.Counter(nltk.bigrams(text))
trigram=collections.Counter(nltk.trigrams([PAD,PAD]+text))

#加一平滑需要的频率
biPast=collections.Counter(h for h,_ in bigram.items())   
triPast=collections.Counter(h for h,_ in trigram.items())

#预测函数(加一平滑和回退)
def predict_next(context,k=5):
    w1,w2=context[-2],context[-1]
    past=(w1,w2)
    Down=triPast.get(past,0)
    Dict={w:trigram[(w1,w2,w)] for w in vocab}

    if max(Dict.values())==0:          
        Down=biPast.get(w2,0)
        Dict={w:biPast[(w2, w)] for w in vocab}

        if max(Dict.values())==0:      
            Dowm=len(text)
            Dict=unigram

    scores={w:(c+1)/(Down+vocab_size) for w,c in Dict.items()}
    return sorted(scores,key=scores.get,reverse=True)[:k]

sentence=input("请输入一句话：")
sentence=sentence.strip().lower().split()

k=int(input("请输入想要预测的词数k:"))
print(predict_next(sentence,k=k))

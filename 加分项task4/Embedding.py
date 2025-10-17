import nltk
from gensim.models import Word2Vec

#超参数
vector_size=100
window=5
min_count=2
epochs=10
ngram=1

#取语料
nltk.download('brown',quiet=True)
sents=nltk.corpus.brown.sents()

#词嵌入
model=Word2Vec(sents,vector_size=vector_size,window=window,min_count=min_count,
                 workers=4,epochs=epochs,sg=ngram,seed=42)

#相似词查询
print('与government相似度最高的:',[w for w, _ in model.wv.most_similar('government',topn=5)])
print('king-man+woman=:',[w for w, _ in model.wv.most_similar(positive=['king','woman'],negative=['man'], topn=1)])
print('computer与machine的相似度:',round(model.wv.similarity('computer','machine'),3))

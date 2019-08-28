# Importing the libraries
import numpy as np
import re
import pickle 
import nltk
#import heapq
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import pandas as pd
import csv
from gensim.models import Word2Vec


#Data=pd.read_excel('Restaurant - Copy.xlsx')
Data = pd.read_csv('Restaurant-Copy.csv')

Data= Data[Data['Polarity'] != 'neutral']
Data= Data[Data['Polarity'] != 'conflict']
z=Data['Polarity'].values.astype('str')

#dividing data into depandent and indepandent variable
X=Data.iloc[:,1].values
y=Data.iloc[:,3].values


for data in range(len(y)):
    if(y[data]=='positive'):
        y[data]=1
    elif(y[data]=='negative'):
        y[data]=0


'''
#Storing as pickle Files
with open('x.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#Umpickling the dataset
with open('x.pickle','rb') as f:
    X=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)

'''

#opening stopwords
with open('stopword_text_bn.txt', 'r', encoding='utf8') as bn:
    bangla_stop_words = [line.strip() for line in bn]


#bangla unicode for data preprocess
whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
bangla_digits = u"[\u09E6\u09E7\u09E8\u09E9\u09EA\u09EB\u09EC\u09ED\u09EE\u09EF]+"
english_chars = u"[a-zA-Z0-9]"
punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
bangla_fullstop = u"\u0964"     #bangla fullstop(dari)
punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"


corpus = []

#preprocessing data
def preprocessing_bangla_text(text):
    text = re.sub(bangla_digits, " ", text)
    text = re.sub(punc, " ", text)
    text = re.sub(english_chars, " ", text)
    text = re.sub(bangla_fullstop, " ", text)
    text = re.sub(punctSeq, " ", text)        
    text = whitespace.sub(" ", text).strip()  #removing multiple whitespace
    text = ' '.join([word for word in text.split() if word not in bangla_stop_words])  # remove stop words
    corpus.append(text) 



for dt in X:
    cleaned_text = preprocessing_bangla_text(dt)
    


#creating word_tokenize    
words = []
for i in range(len(corpus)):
    words.append(corpus[i].split())
    
    
sent_len = len(words)
#max([len(t.split()) for t in texts])    
max_len = max([len(word.split()) for word in corpus]) 
    
#size = dimention feature vector
#worker = is used for more fast processing such as parallel tradding 
    
#create CBOW
model_1 = Word2Vec(words, size = 100, window = 3, min_count = 1, workers = 3)
word_dict = list(model_1.wv.vocab)
unique_word_len = len(word_dict) +1
#vocab_size = len(model_1.wv.vocab)

file_name = 'sentiment_word2vec.txt'
model_1.wv.save_word2vec_format(file_name, binary = False)

#model save
#model_1.wv.save_word2vec_format('model_1.bin')

# another saving formate in text formante
#model_1.wv.save_word2vec_format('model_1.bin', binary = False)

#Load model again
#model_1_load = gensim.models.Word2Vec.load('model_1.bin')

#output_files = 'model_1.bin'
#model = gensim.models.Word2Vec.load(output_files)

#create skip-gram
model_2 = Word2Vec(words, size=300, window=5, min_count=2, workers=40,seed=1,sample=1e-5,hs=1,negative=5,iter=30, sg = 1)
word_dict_skip = list(model_2.wv.vocab)
vocab_size = len(model_2.wv.vocab)



# vector representation of word
vector = model_2.wv['স্টাফ']

#most similar word
similar = model_2.wv.most_similar('স্টাফ')

in_list = []
for item in corpus:
    if 'স্টাফ'  in item:
        in_list.append(item)
        


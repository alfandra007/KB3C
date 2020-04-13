# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:57:19 2020

@author: rezas
"""
# In[]:
import gensim
genmod = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
# In[]:
genmod['love']
# In[]:
genmod['faith']
# In[]:
genmod['fall']
# In[]:
genmod['sick']
# In[]:
genmod['clear']
# In[]:
genmod['shine']
# In[]:
genmod['bag']
# In[]:
genmod['car']
# In[]:
genmod['wash']
# In[]:
genmod['motor']
# In[]:
genmod['cycle']
# In[]:
genmod.similarity('wash', 'clear')
# In[]:
genmod.similarity('bag', 'love')
# In[]:
genmod.similarity('motor', 'car')
# In[]:
genmod.similarity('sick', 'faith')
# In[]:
genmod.similarity('cycle', 'shine')
# In[]:
import re
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) #hapus tag html
    sent = re.sub(r'(\w)\'(\w)', ' ', sent) #hapus petik satu
    sent = re.sub(r'\W', ' ', sent) #hapus tanda baca
    sent = re.sub(r'\s+', ' ', sent) #hapus spasi yang berurutan
    return sent.split()
# In[]:
import random
class PermuteSentences(object):
    def __init__(self, sents):
        self.sents = sents
        
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent
# In[]:
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# In[]:
import os
unsup_sentences = []
# In[]:
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("D:/aclImdb/"+dirname)):
        if fname[-4:] == '.txt':
            with open("D:/aclImdb/"+dirname+"/"+fname,encoding='UTF-8') as f:
                sent = f.read()
                words = extract_words(sent)
                unsup_sentences.append(TaggedDocument(words,[dirname+"/"+fname]))
# In[]:
for dirname in ["txt_sentoken/pos", "txt_sentoken/neg"]:
    for fname in sorted(os.listdir("D:/aclImdb/"+dirname)):
        if fname[-4:] == '.txt':
            with open("D:/aclImdb/"+dirname+"/"+fname,encoding='UTF-8') as f:
                for i, send in enumerate(f):
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(words,["%s/%s-%d" % (dirname, fname, i)]))
# In[]:
with open("D:/stanforSentimentTreebank/original_rt_snippets.txt", encoding='UTF-8') as f:
    for i, sent in enumerate(f):
        words = extract_words(sent)
        unsup_sentences.append(TaggedDocument(words,["rt-%d" % i]))
# In[]:
import re
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) #hapus tag html
    sent = re.sub(r'(\w)\'(\w)', ' ', sent) #hapus petik satu
    sent = re.sub(r'\W', ' ', sent) #hapus tanda baca
    sent = re.sub(r'\s+', ' ', sent) #hapus spasi yang berurutan
    return sent.split()

import random
class PermuteSentences(object):
    def __init__(self, sents):
        self.sents = sents
        
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent
# In[]:
mute=PermuteSentences(unsup_sentences)
# In[]:
model = Doc2Vec(mute, dm=0, hs=1, vector_size=50)
# In[]:
model.delete_temporary_training_data(keep_inference=True)
            
# In[]:
model.delete_temporary_training_data(keep_inference=True)
model.save('reza.d2v')
# In[]:
model.infer_vector(extract_words("i will go home"))
# In[]:
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([model.infer_vector(extract_words("she going to school, after wash hand"))], 
                  [model.infer_vector(extract_words("Service sucks2."))])
# In[]:
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([model.infer_vector(extract_words("Jangan lupa cuci tangan"))], 
                  [model.infer_vector(extract_words("Pake masker juga"))])

# In[]:
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))
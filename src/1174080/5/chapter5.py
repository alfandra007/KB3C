# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:59:21 2020

@author: Handi
"""

#%% Soal no 1

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
handi_model = gensim.models.KeyedVectors.load_word2vec_format('C:/xampp/cgi-bin/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)

#%%
handi_model.similarity('wash', 'clear')
#%%
handi_model.similarity('bag', 'love')
#%%
handi_model.similarity('motor', 'car')
#%%
handi_model.similarity('sick', 'faith')
#%%
handi_model.similarity('cycle', 'shine')
#%%
handi_model['love']
#%%
handi_model['faith']
#%%
handi_model['fall']
#%%
handi_model['sick']
#%%
handi_model['clear']
#%%
handi_model['shine']
#%%
handi_model['bag']
#%%
handi_model['car']
#%%
handi_model['wash']
#%%
handi_model['motor']
#%%
handi_model['cycle']
#%% Soal no 2

import re 
    
test_string = "teast 1"
print ("test 2" +  test_string) 
res = re.findall(r'\w+', test_string) 

print ("ok" +  str(res))  # mengecek apakah ada kata "algoritmatik" di kamus

#%%
    
import random

sent_matrix = [ ['1', '2'],
                ['3', '4']
              ]

result = "ok"
for elem in sent_matrix:
    result += random.choice(elem) + " "

print (result)

#%% Soal no 3

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


doc = ["test 1"]

tokenized_doc = ['ok']
tokenized_doc

print(doc)

#%%
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
tagged_data

model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)

model.save("test_doc2vec.model")

model= Doc2Vec.load("test_doc2vec.model")

model.wv.vocab


#%% Soal no 4
import re
import os
unsup_sentences = []

for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("aclImdb/" + dirname)):
        if fname[-4:] == '.txt':
            with open("aclImdb/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = (sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))
                

#%% soal no 5
                
mute = (unsup_sentences)

model.delete_temporary_training_data(keep_inference=True)
                
                
#%% soal no 6                
             
model.save('handi.d2v')


model.delete_temporary_training_data(keep_inference=True)             

#%% soal no 7

model.infer_vector(extract_words(""))

#%% soal no 8

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(extract_words("!"))])

#%% soal no 9

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))
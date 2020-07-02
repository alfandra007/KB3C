# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:43:41 2020

@author: USER
"""

#%% Soal no 1

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
izza_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)


#%%
izza_model['love']
#%%
izza_model['faith']
#%%
izza_model['fall']
#%%
izza_model['sick']
#%%
izza_model['clear']
#%%
izza_model['shine']
#%%
izza_model['bag']
#%%
izza_model['car']
#%%
izza_model['wash']
#%%
izza_model['motor']
#%%
izza_model['cycle']
#%%
izza_model.similarity('wash', 'clear')
#%%
izza_model.similarity('bag', 'love')
#%%
izza_model.similarity('motor', 'car')
#%%
izza_model.similarity('sick', 'faith')
#%%
izza_model.similarity('cycle', 'shine')


#%% Soal no 2
import re 
    
test_string = "Buah Apel,    Makanan Sehat dan Bervitamin"
print ("Sangat bagus di komsumsi : " +  test_string) 
res = re.findall(r'\w+', test_string) 

print ("The list of words is : " +  str(res)) 

#%%
    
import random

sent_matrix = [ ['Buah', 'Apel'],
                ['Mengandung', 'Vitamin'],
                ['A', 'Dan'],
                ['Mengandung', 'Kalium']
              ]

result = ""
for elem in sent_matrix:
    result += random.choice(elem) + " "

print (result)


#%% Soal no 3
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

## Exapmple document (list of sentences)
doc = ["I love holidays",
        "I love shopping",
        "I love Indonesia",
        "This is a good day",
        "This is a good Things",
        "This is a good Handphone"]

tokenized_doc = ['love']
tokenized_doc

print(doc)

#%%
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
tagged_data
## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
# Save trained doc2vec model
model.save("test_doc2vec.model")
## Load saved doc2vec model
model= Doc2Vec.load("test_doc2vec.model")
## Print model vocabulary
model.wv.vocab


#%% Soal no 4
import re
import os
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("aclImdb/" + dirname)):
        if fname[-4:] == '.txt':
            with open("aclImdb/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = (sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))
                
                
#%% soal no 5         
#Pengacakan data
mute = (unsup_sentences)

#Pembersihan data
model.delete_temporary_training_data(keep_inference=True)
                
                
#%% soal no 6                
#Save data
model.save('izzah.d2v')

#Delete temporary data
model.delete_temporary_training_data(keep_inference=True)             

#%% soal no 7

model.infer_vector(extract_words("Selamat Beraktivitas"))

#%% soal no 8

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(extract_words("Semoga hari kalian menyenangkan"))],
        [model.infer_vector(extract_words("Hari ini hari yang baik"))])

#%% soal no 9

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))



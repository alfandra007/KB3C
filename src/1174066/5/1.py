# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:52:17 2020

@author: Rin
"""
#%% Soal 1
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_dirga = gensim.models.KeyedVectors.load_word2vec_format('N:/KB/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

#%%
model_dirga['love']
#%%
model_dirga['faith']
#%%
model_dirga['fall']
#%%
model_dirga['sick']
#%%
model_dirga['clear']
#%%
model_dirga['shine']
#%%
model_dirga['bag']
#%%
model_dirga['car']
#%%
model_dirga['wash']
#%%
model_dirga['motor']
#%%
model_dirga['cycle']
#%%
model_dirga.similarity('wash', 'clear')
#%%
model_dirga.similarity('bag', 'love')
#%%
model_dirga.similarity('motor', 'car')
#%%
model_dirga.similarity('sick', 'faith')
#%%
model_dirga.similarity('cycle', 'shine')

#%% Soal 2
import re 
    
test_string = "Dirga Brajamusti, adalah nama aku"
print ("Faktanya: " +  test_string) 
res = re.findall(r'\w+', test_string) 
print ("The list of words is : " +  str(res)) 
#%% 
import random
sent_matrix = [ ['Ini', 'Data'], ['Untuk', 'Merandom'], ['Isi', 'Yang'], ['Ada', 'Disini']]
result = ""
for elem in sent_matrix:
    result += random.choice(elem) + " "
print (result)
#%% Soal 3
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
## Exapmple document (list of sentences)
doc = ["I love machine learning",
        "I love coding in python",
        "This is a good pc",
        "This is a good mac",
        "This is a good phone"]
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


#%% Soal 4
import re
import os
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("N:/KB/aclImdb/" + dirname)):
        if fname[-4:] == '.txt':
            with open("N:/KB/aclImdb/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = (sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))
                

#%% soal 5    
#Pengacakan data
mute = (unsup_sentences)

#Pembersihan data
model.delete_temporary_training_data(keep_inference=True)
                
                
#%% soal 6
#Save data
model.save('dirga.d2v')

#Delete temporary data
model.delete_temporary_training_data(keep_inference=True)             

#%% soal 7
model.infer_vector(res)

#%% soal 8
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(["datanya", "banyak"])],
        [model.infer_vector(["pusing", "data"])])

#%% soal 9
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))
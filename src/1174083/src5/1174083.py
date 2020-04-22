# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:54:55 2020

@author: Bakti Qilan
"""
# In[Praktek no. 1]
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
baktiqilan_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

# In[]
baktiqilan_model['love']
# In[]
baktiqilan_model['faith']
# In[]
baktiqilan_model['fall']
# In[]
baktiqilan_model['sick']
# In[]
baktiqilan_model['clear']
# In[]
baktiqilan_model['shine']
# In[]
baktiqilan_model['bag']
# In[]
baktiqilan_model['car']
# In[]
baktiqilan_model['wash']
# In[]
baktiqilan_model['motor']
# In[]
baktiqilan_model['cycle']
# In[]
baktiqilan_model.similarity('wash', 'clear')
# In[]
baktiqilan_model.similarity('bag', 'love')
# In[]
baktiqilan_model.similarity('motor', 'car')
# In[]
baktiqilan_model.similarity('sick', 'faith')
# In[]
baktiqilan_model.similarity('cycle', 'shine')
# In[Praktek no. 2]
import re 
    
test_string = "Mawar itu merah, violet itu biru.!!!"
print ("Hasil : " +  test_string) 
res = re.findall(r'\w+', test_string) 

print ("The list of words is : " +  str(res)) 

# In[]
import random

sent_matrix = [ ['Mawar', 'itu'],
                ['hitam', 'violet'],
                ['itu', 'merah'],
                ['kuning', 'hijau']
              ]
result = ""
for elem in sent_matrix:
    result += random.choice(elem) + " "

print (result)

# In[Praktek no. 3]

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
## Exapmple document (list of sentences)
doc = ["I love pdf",
        "I love u",
        "I love sleep",
        "This is a good mouse",
        "This is a good house",
        "This is a good pause"]

tokenized_doc = ['love']
tokenized_doc

print(doc)
# In[]
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

# In[Praktek no. 4]
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
                
# In[Praktek no. 5] 
#Pengacakan data
mute = (unsup_sentences)

#Pembersihan data
model.delete_temporary_training_data(keep_inference=True)
                
# In[Praktek no. 6]               
#Save data
model.save('baktiqilan.d2v')

#Delete temporary data
model.delete_temporary_training_data(keep_inference=True)             

# In[Praktek no. 7]
model.infer_vector(["copas?", "disini aja"])

# In[Praktek no. 8]
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(["copas?", "boleh"])],
        [model.infer_vector(["plagiat?", "jangan"])])

# In[Praktek no. 9]
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))

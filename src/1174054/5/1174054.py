# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:35:43 2020

@author: Aulyardha Anindita
"""

#%% Nomor 1
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dita = gensim.models.KeyedVectors.load_word2vec_format('D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 5/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

#%%
dita['love']
#%%
dita['faith']
#%%
dita['fall']
#%%
dita['sick']
#%%
dita['clear']
#%%
dita['shine']
#%%
dita['bag']
#%%
dita['car']
#%%
dita['wash']
#%%
dita['motor']
#%%
dita['cycle']
#%%
dita.similarity('wash', 'clear')
#%%
dita.similarity('bag', 'love')
#%%
dita.similarity('motor', 'car')
#%%
dita.similarity('sick', 'faith')
#%%
dita.similarity('cycle', 'shine')

#%% Nomor 2
import re 
test_string = "Aulyardha Anindita,    Kuliah di Politeknik Pos Indonesia"
print ("Biodata : " +  test_string) 
res = re.findall(r'\w+', test_string) 
print ("List kata nya adalah : " +  str(res)) 
#%%
import random
sent_matrix = [ ['Aulyardha', 'Anindita'],
                ['Kuliah', 'di'],
                ['Politeknik', 'Pos'],
                ['Indonesia']
              ]
result = ""
for elem in sent_matrix:
    result += random.choice(elem) + " "
print (result)

#%%
import re
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) #menghapus tag html
    sent = re.sub(r'(\w)\'(\w)', ' ', sent) #menghapus petik satu
    sent = re.sub(r'\W', ' ', sent) #menghapus tanda baca
    sent = re.sub(r'\s+', ' ', sent) #menghapus spasi yang berurutan
    return sent.split()

#%% Nomor 3
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

## Contoh dokumen
doc = ["saya suka membaca",
        "saya suka menulis",
        "saya suka menonton",
        "saya sering tersenyum",
        "saya sering jalan-jalan",
        "saya sering makan"]
tokenized_doc = ['sering']
tokenized_doc
print(doc)

#%%
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
tagged_data
# mengtrain doc2vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
# menyimpan trained doc2vec model
model.save("test_doc2vec.model")
# meload doc2vec model
model= Doc2Vec.load("test_doc2vec.model")
#menampilkan model
model.wv.vocab

#%% Nomor 4
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
                
#%% Nomor 5
mute = (unsup_sentences) #mengacak data

model.delete_temporary_training_data(keep_inference=True) #membersihkan data

#%% Nomor 6              
                
model.save('dita.d2v') #menyimpan data

model.delete_temporary_training_data(keep_inference=True) #menghapus temporary data

#%% Nomor 7

model.infer_vector(extract_words("Aulyardha Anindita"))

#%% Nomor 8

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(extract_words("Dita pulang kampung"))],
        [model.infer_vector(extract_words("Karena kampus diliburkan gara-gara corona"))])

#%% Nomor 9

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
aulyta = datasets.load_diabetes()
X = aulyta.data[:150]
y = aulyta.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))

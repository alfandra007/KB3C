# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:26:38 2020

@author: Handi
"""

#%% Soal 1
import pandas as pd #digunakan untuk mengimport library pandas dengan alias pd
pd = pd.read_csv("csv.csv") #membaca file csv 

#%% Soal 2
love=pd[:450] #membagi data training menjadi 450
fixs=pd[450:] #membagi data menjadi 450 atau sisa dari data yang tersedia

#%% Soal 3
import pandas as hans #untuk import library pandas berguna untuk mengelola dataframe
hans = hans.read_csv("Youtube03-LMFAO.csv") #membaca file dengan format csv

spam=hans.query('CLASS == 1') #membagi tabel spam
nospam=hans.query('CLASS == 0')#membagi tabel no spam

from sklearn.feature_extraction.text import CountVectorizer #untuk import countvectorizer berfungsi untuk memecah data tersebut menjadi sebuah kata yang lebih sederhana
vectorizer = CountVectorizer () #ntuk menjalankan fungsi tersebut, pada code ini tidak ada hasilnya dikarenakan spyder tidak mendukung hasil dari instasiasi.

dvec = vectorizer.fit_transform(hans['CONTENT']) #untuk melakukan pemecahan data pada dataframe yang terdapat pada kolom konten
dvec #Untuk menampilkan hasil dari code sebelumnya

Daptarkata= vectorizer.get_feature_names()

dshuf = hans.sample(frac=1)

love=dshuf[:300]
fixs=dshuf[300:]

love_att = vectorizer.fit_transform(love['CONTENT'])
love_att

love_label=love['CLASS']
fixs_label=fixs['CLASS']

#%% Soal 4

from sklearn import svm
clfsvm = svm.SVR(gamma = 'auto')
clfsvm.fit(love_att, love_label)

#%%soal 5

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(love_att, love_label)

#%%soal 6

from sklearn.metrics import confusion_matrix
pred_labels=clftree.predict(fixs)
cm=confusion_matrix(fixs_label,pred_labels)

#%%

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

#%%soal 7
    
from sklearn.model_selection import cross_val_score

scores=cross_val_score(clftree,love_att,love_label,cv=5)

skor_rata2=scores.mean()
skoresd=scores.std()

#%%soal 8 

max_features_opts = range(5, 50, 5) #max_features_opts sebagai variabel untuk membuat range 5,50,5
n_estimators_opts = range(10, 200, 20) #n_estimators_opts sebagai variabel untuk membuat range 10,200,20
rf_params = hans.empty((len(max_features_opts)*len(n_estimators_opts),4), float) #rf_params sebagai variabel untuk menjumlahkan yang sudah di tentukan sebelumnya
i = 0
for max_features in max_features_opts: #pengulangan 
    for n_estimators in n_estimators_opts: #pengulangan
        clftree = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators) #menampilkan variabel csf
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) #scores sebagai variabel training 
        rf_params[i,0] = max_features #index 0
        rf_params[i,1] = n_estimators #index 1
        rf_params[i,2] = scores.mean() #index 2
        rf_params[i,3] = scores.std() * 2 #index 3
        i += 1 #dengan ketentuan i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %(max_features, n_estimators, scores.mean(), scores.std() * 2))
        #print hasil pengulangan yang sudah ditentukan
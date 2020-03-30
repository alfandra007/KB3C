# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:29:30 2020

@author: TIA
"""

import pandas as pd #mengimport library pandas dan menamainya pd
#%%
tia = pd.read_csv("D:/TI/SMT 6/AI/Chapter4/New folder/src/1174086.csv") #membuat variable bernama tia dan mengisinya dengan data dari dataset dummy yang telah dibuat
a = tia.head() #untuk melihat 5 baris pertama dari data tia
tia.shape #untuk mengetahui berapa banyak baris data
print(a) #menampilkan isi dari varibale a pada console
#%%
dtra = tia[:450] #memasukkan 450 data pertama ke dalam variable dtra
dtes = tia[450:] #memasukkan 50 data terakhir kedalam variable dtes
#%% memasukkan data dari file csv tersebut ke dalam variable data
data=pd.read_csv("D:/TI/SMT 6/AI/Chapter4/New folder/src/1174086.csv")
spam=data.query('CLASS == 1')
nospam=data.query('CLASS == 0')
#%% melakukan fungsi bag of word dengan cara menghitung semua kata
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
#%% melakukan bag of word pada dataframe pada colom CONTENT
data_vektorisasi = vectorizer.fit_transform(data['CONTENT'])
#%% melihat isi vektorisasi
data_vektorisasi
#%% melihat isi data pada baris ke 345
print(data['CONTENT'][345])
#%% untuk mengambil apa saja nama kolom yang tersedia
dk=vectorizer.get_feature_names()
#%%: melakukan randomisasi agar hasil sempurna pada saat klasifikasi
dshuf = data.sample(frac=1)
#%%: membuat data traning dan testing
dk_train=dshuf[:300]
dk_test=dshuf[300:]
#%%: melakukan training pada data training dan di vektorisasi
dk_train_att=vectorizer.fit_transform(dk_train['CONTENT'])
print(dk_train_att)
#%% melakukan testing pada data testing dan di vektorisasi
dk_test_att=vectorizer.transform(dk_test['CONTENT'])
print(dk_test_att)
#%%: Dimana akan mengambil label spam dan bukan spam
dk_train_label=dk_train['CLASS']
print(dk_train_label)
dk_test_label=dk_test['CLASS']
print(dk_test_label)
#%%

from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(dk_train_att, dk_train_label)
clfsvm.score(dk_test_att, dk_test_label)
#%%
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(dk_train_att, dk_train_label)
clftree.score(dk_test_att, dk_test_label)

#%%
from sklearn.metrics import confusion_matrix
pred_labels = clftree.predict(dk_test_att)
cm = confusion_matrix(dk_test_label, pred_labels)
cm

#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree,dk_train_att,dk_train_label,cv=5)
scorerata2=scores.mean()
scorersd=scores.std()
#%%:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95
#% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),
scores.std() * 2))
#%%:
scorestree = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(),
scorestree.std() * 2))
#%%:
scoressvm = cross_val_score(clfsvm, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(),
scoressvm.std() * 2))


#%%
import numpy as np
max_features_opts = range(1, 10, 1)
n_estimators_opts = range(2, 40, 4)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4) , float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, dk_train_att, dk_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)"
% (max_features, n_estimators, scores.mean(), scores.std() * 2))


#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.clf()
ax = fig.gca(projection='3d')
x = rf_params[:,0]
y = rf_params[:,1]
z = rf_params[:,2]
ax.scatter(x, y, z)
ax.set_zlim(0.6, 1)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()

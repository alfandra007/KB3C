# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:26:01 2020

@author: Rin
"""
#%% Soal 1
import pandas as pd #digunakan untuk mengimport library pandas dengan alias pd
pd = pd.read_csv("N:/Tugas/Kuliah/Semester 6/Kecerdasan Buatan/KB3C Ngerjain/src/1174066/4/csv.csv") #membaca file csv 

#%% Soal 2
d_train=pd[:450] #membagi data training menjadi 450
d_test=pd[450:] #membagi data menjadi 50 atau sisa dari data yang tersedia

#%% Soal 3
import pandas as pd #digunakan untuk mengimport library pandas dengan alias pd
d = pd.read_csv("N:/Tugas/Kuliah/Semester 6/Kecerdasan Buatan/KB3C Ngerjain/src/1174066/4/Youtube04-Eminem.csv") #Membaca file csv

from sklearn.feature_extraction.text import CountVectorizer #import fungsi countvectorize dari sklearn
vectorizer = CountVectorizer() #membuat instansi CountVectorizer

dvec = vectorizer.fit_transform(d['CONTENT']) #Memasukkan data ke dvec
dvec #Melihat data yang dimasukkan ke dvec

daptarkata = vectorizer.get_feature_names() #Mendapatkan data dan memasukkannya ke daptarkata

dshuf = d.sample(frac=1) #Memasukkan sample kedalam variable dshuf

d_train = dshuf[:300] #Membuat data training
d_test = dshuf[300:] #Membuat data test

d_train_att = vectorizer.fit_transform(d_train['CONTENT']) #Memasukkan data training dari vectorizer
d_train_att #Melihat data training

d_test_att = vectorizer.transform(d_test['CONTENT']) #Memasukkan data test dari vectorizer
d_test_att #Melihat data training

d_train_label = d_train['CLASS'] #Memberi label
d_test_label = d_test['CLASS'] #Memberi Label

#%% Soal 4
# SVM
from sklearn import svm #Mengimport svm dari sklearn
clfsvm = svm.SVC() #Membuat svc kedalam variable svm
clfsvm.fit(d_train_att, d_train_label) #Memprediksi data dari data training
clfsvm.score(d_test_att, d_test_label) #Memunculkan clf sebagai testing yang sudah di training tadi


#%% Soal 5
# Decission Tree
from sklearn import tree #Mengimport tree dari sklearn
clftree = tree.DecisionTreeClassifier() #Membuat decision tree
clftree.fit(d_train_att, d_train_label) #Memprediksi data dari data training
clftree.score(d_test_att, d_test_label) #Memunculkan clf sebagai testing yang sudah di training tadi

#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)
clf.fit(d_train_att, d_train_label)
clf.score(d_test_att, d_test_label)

#%% Soal 6
# Confusion Matrix - Decission Tree
from sklearn.metrics import confusion_matrix
pred_labelstree = clftree.predict(d_test_att)
cmtree = confusion_matrix(d_test_label, pred_labelstree)
cmtree

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

types = pd.read_csv("N:/zz/KB3A-master/src/1174006/chapter4/classes.txt",sep='\s+', header=None,usecols=[1], names=['type'])
types = types['type']
types

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4), dpi=100)
plot_confusion_matrix(cmtree, classes=types, normalize=True)
plt.show()

#%%
# Confusion Matrix - Random Forest
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
cm = confusion_matrix(d_test_label, pred_labels)

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

types = pd.read_csv("classes.txt",sep='\s+', header=None,usecols=[1], names=['type'])
types = types['type']
types

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4), dpi=100)
plot_confusion_matrix(cmtree, classes=types, normalize=True)
plt.show()
#%% Soal 7
# Cross Validation
from sklearn.model_selection import cross_val_score

scorestree = cross_val_score(clftree, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

scoressvm = cross_val_score(clfsvm, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#%% Soal 8
# Komponen Informasi
max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" % (max_features, n_estimators, scores.mean(), scores.std() * 2))

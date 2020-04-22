# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:56:17 2020

@author: Aulyardha Anindita
"""

# In[]:
print(1174054%4)

# In[]:
# Nomor 1
import pandas as pd #import package pandas, lalu dialiaskan menjadi pd.
sp = pd.read_csv('D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 4/datadummy.csv', delimiter = ',') #membaca file csv dimana data pada file csv dipisahkan oleh koma, lalu ditampung di variable sp.

# In[]:
# Nomor 2
sp1, sp2 = sp[:450], sp[450:] #membagi data menjadi dua bagian, variable sp1 untuk menampung 450 baris data pertama, variable sp2 untuk menampung 50 baris data terakhir.

# In[]:
# Nomor 3 Vektorisasi Data
import pandas as pd
d = pd.read_csv("D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 4/Youtube04-Eminem.csv")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

dvec = vectorizer.fit_transform(d['CONTENT'])
dvec

daptarkata = vectorizer.get_feature_names()

dshuf = d.sample(frac=1)

d_train = dshuf[:300]
d_test = dshuf[300:]

d_train_att = vectorizer.fit_transform(d_train['CONTENT'])
d_train_att

d_test_att = vectorizer.transform(d_test['CONTENT'])
d_test_att

d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']

# In[]:
# Decission Tree
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(d_train_att, d_train_label)
clftree.score(d_test_att, d_test_label)

# In[]:
# SVM
from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(d_train_att, d_train_label)
clfsvm.score(d_test_att, d_test_label)

# In[]:
# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)
clf.fit(d_train_att, d_train_label)
clf.score(d_test_att, d_test_label)

# In[]:
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

types = pd.read_csv("D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 4/classes.txt",sep='\s+', header=None,usecols=[1], names=['type'])
types = types['type']
types

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4), dpi=100)
plot_confusion_matrix(cmtree, classes=types, normalize=True)
plt.show()

# In[]:
# Confusion Matrix - SVM
from sklearn.metrics import confusion_matrix
pred_labelssvm = clfsvm.predict(d_test_att)
cmsvm = confusion_matrix(d_test_label, pred_labelssvm)
cmsvm

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

types = pd.read_csv("D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 4/classes.txt",sep='\s+', header=None,usecols=[1], names=['type'])
types = types['type']
types

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4), dpi=100)
plot_confusion_matrix(cmsvm, classes=types, normalize=True)
plt.show()

# In[]:
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

types = pd.read_csv("D:/Mata Kuliah/Tingkat 3/Semester 6/Kecerdasan Buatan/Chapter 4/classes.txt",sep='\s+', header=None,usecols=[1], names=['type'])
types = types['type']
types

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4), dpi=100)
plot_confusion_matrix(cmtree, classes=types, normalize=True)
plt.show()

# In[]:
# Cross Validation
from sklearn.model_selection import cross_val_score

scorestree = cross_val_score(clftree, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

scoressvm = cross_val_score(clfsvm, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# In[]:
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
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %               (max_features, n_estimators, scores.mean(), scores.std() * 2))

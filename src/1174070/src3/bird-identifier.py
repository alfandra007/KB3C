# -*- coding: utf-8 -*-
"""
Created on Thu Mar  11 22:18:40 2020

@author: Bakti Qilan
"""
# In[4]: Random Forest
import pandas as pd #import library pandas sebagai pd

imgatt = pd.read_csv("E:/backup/sem 6/Kecerdasan Buatan/CUB_200_2011/attributes/image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0,1,2], names=['imgid', 'attid', 'present'])#untuk membaca file txt

# In[4.1]:

imgatt.head()

# In[4.2]:

imgatt.shape

# In[4.3]:

imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present')

# In[4.4]:

imgatt2.head()

# In[4.5]:

imgatt2.shape

# In[4.5]:

imglabels = pd.read_csv("E:/backup/sem 6/Kecerdasan Buatan/CUB_200_2011/image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label'])

imglabels = imglabels.set_index('imgid')

# In[4.6]:

imglabels.head()

# In[4.7]:

imglabels.shape

# In[4.8]:

df = imgatt2.join(imglabels)
df = df.sample(frac=1)

# In[4.9]:

df_att = df.iloc[:, :312]
df_label = df.iloc[:, 312:]

# In[4.10]:

df_att.head()

# In[4.11]:

df_label.head()

# In[4.12]:

df_train_att = df_att[:8000]
df_train_label = df_label[:8000]
df_test_att = df_att[8000:]
df_test_label = df_label[8000:]

df_train_label = df_train_label['label']
df_test_label = df_test_label['label']

# In[4.13]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)

# In[4.14]:

clf.fit(df_train_att, df_train_label)

# In[4.15]:

print(clf.predict(df_train_att.head()))

# In[4.16]:

clf.score(df_test_att, df_test_label)

# In[5]: Confusion Matrix

from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(df_test_att)
cm = confusion_matrix(df_test_label, pred_labels)

# In[5.1]: Confusin Matrix

cm

# In[5.2]: Confusin Matrix
# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# In[5.3]: Confusin Matrix

birds = pd.read_csv("E:/backup/sem 6/Kecerdasan Buatan/CUB_200_2011/classes.txt",
                    sep='\s+', header=None, usecols=[1], names=['birdname'])
birds = birds['birdname']
birds

# In[5.4]: Confusin Matrix

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(60,60), dpi=300)
plot_confusion_matrix(cm, classes=birds, normalize=True)
plt.savefig("E:/backup/sem 6/Kecerdasan Buatan/KB3C - Copy/figures/1174083/figures3/ganti.png")

# In[6]: Decission Tree dan SVM

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(df_train_att, df_train_label)
clftree.score(df_test_att, df_test_label)

# In[6.1]: Decission Tree dan SVM

from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(df_train_att, df_train_label)
clfsvm.score(df_test_att, df_test_label)

# In[7]: Cross Validaiton

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# In[7.1]: Cross Validaiton

scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

# In[7.2]: Cross Validaiton

scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

# In[8]: Pengamatan Komponen Informasi

max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %               (max_features, n_estimators, scores.mean(), scores.std() * 2))

# In[8.1]: Pengamatan Komponen Informasi

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
ax.set_zlim(0.2, 0.5)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()

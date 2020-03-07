# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:45:40 2020

@author: Bakti Qilan
"""
# In[0]:

1174083 % 3
#Hasilnya = 0 maka mengambil variable dengan nama makanan

# In[1]:
# load dataset (menggunakan student-mat)
import pandas as pd
mochi = pd.read_csv('E://backup/sem 6/Kecerdasan Buatan/KB3C - Copy/src/1174083/src2/dataset/student-mat.csv', sep=';')
len(mochi)

# In[2]:
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
mochi['pass'] = mochi.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
mochi = mochi.drop(['G1', 'G2', 'G3'], axis=1)
mochi.head()

# In[3]:
# use one-hot encoding on categorical columns
mochi = pd.get_dummies(mochi, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
mochi.head()

# In[4]:
# shuffle rows
mochi = mochi.sample(frac=1)
# split training and testing data
mochi_train = mochi[:500]
mochi_test = mochi[500:]

mochi_train_att = mochi_train.drop(['pass'], axis=1)
mochi_train_pass = mochi_train['pass']

mochi_test_att = mochi_test.drop(['pass'], axis=1)
mochi_test_pass = mochi_test['pass']

mochi_att = mochi.drop(['pass'], axis=1)
mochi_pass = mochi['pass']

# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(mochi_pass), len(mochi_pass), 100*float(np.sum(mochi_pass)) / len(mochi_pass)))


# In[5]:
# fit a decision tree
from sklearn import tree
cilok = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
cilok = cilok.fit(mochi_train_att, mochi_train_pass)

# In[6]:
# visualize tree
import os
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/Library/bin/graphviz/'
import graphviz
donat = tree.export_graphviz(cilok, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(mochi_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True)
kue = graphviz.Source(donat)
kue

# In[7]:
# save tree
tree.export_graphviz(cilok, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(mochi_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True)

# In[8]:

cilok.score(mochi_test_att, mochi_test_pass)

# In[9]:
from sklearn.model_selection import cross_val_score
biskuit = cross_val_score(cilok, mochi_att, mochi_pass, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (biskuit.mean(), biskuit.std() * 2))


# In[10]:
for max_depth in range(1, 20):
    cilok = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    biskuit = cross_val_score(cilok, mochi_att, mochi_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, biskuit.mean(), biskuit.std() * 2))

# In[11]:
depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    cilok = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    biskuit = cross_val_score(cilok, mochi_att, mochi_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = biskuit.mean()
    depth_acc[i,2] = biskuit.std() * 2
    i += 1
depth_acc


# In[12]:
import matplotlib.pyplot as plt
fig, puding = plt.subplots()
puding.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()

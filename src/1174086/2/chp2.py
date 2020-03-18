# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# In[0]:

1174086 % 3
#Hasilnya = 0 maka mengambil variable dengan nama makanan

# In[1]:
# load dataset (menggunakan student-mat)
import pandas as pd
tahu = pd.read_csv('D://TI/SMT 6/AI/Chapter 2/KB3C-master - Copy/src/1174086/2/dataset/student-mat.csv', sep=';')
len(tahu)

# In[2]:
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
tahu['pass'] = tahu.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
tahu = tahu.drop(['G1', 'G2', 'G3'], axis=1)
tahu.head()

# In[3]:
# use one-hot encoding on categorical columns
tahu = pd.get_dummies(tahu, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
tahu.head()

# In[4]:
# shuffle rows
tahu = tahu.sample(frac=1)
# split training and testing data
tahu_train = tahu[:500]
tahu_test = tahu[500:]

tahu_train_att = tahu_train.drop(['pass'], axis=1)
tahu_train_pass = tahu_train['pass']

tahu_test_att = tahu_test.drop(['pass'], axis=1)
tahu_test_pass = tahu_test['pass']

tahu_att = tahu.drop(['pass'], axis=1)
tahu_pass = tahu['pass']

# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(tahu_pass), len(tahu_pass), 100*float(np.sum(tahu_pass)) / len(tahu_pass)))


# In[5]:
# fit a decision tree
from sklearn import tree
tempe = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
tempe = tempe.fit(tahu_train_att, tahu_train_pass)

# In[6]:
# visualize tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Tia/Anaconda3/Library/bin/graphviz/'
import graphviz
oncom = tree.export_graphviz(tempe, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(tahu_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True)
makanan = graphviz.Source(oncom)
makanan

# In[7]:
# save tree
tree.export_graphviz(tempe, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(tahu_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True)

# In[8]:

tempe.score(tahu_test_att, tahu_test_pass)

# In[9]:
from sklearn.model_selection import cross_val_score
perkedel = cross_val_score(tempe, tahu_att, tahu_pass, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (perkedel.mean(), perkedel.std() * 2))


# In[10]:
for max_depth in range(1, 20):
    tempe = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    perkedel = cross_val_score(tempe, tahu_att, tahu_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, perkedel.mean(), perkedel.std() * 2))

# In[11]:
depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    tempe = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    perkedel = cross_val_score(tempe, tahu_att, tahu_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = perkedel.mean()
    depth_acc[i,2] = perkedel.std() * 2
    i += 1
depth_acc


# In[12]:
import matplotlib.pyplot as plt
fig, burger = plt.subplots()
burger.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()

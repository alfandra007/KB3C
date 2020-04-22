
# coding: utf-8

buah =  1174076 % 3
print("Hasilnya : ", buah)

# In[1]:

# Banyaknya baris atau panjang
import pandas as pd
gomuGomuNomi = pd.read_csv('dataset/student-mat.csv', sep=';')
len(gomuGomuNomi)

# In[2]:
# Sumbu 0 = kolom, 1 = baris
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
gomuGomuNomi['pass'] = gomuGomuNomi.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
gomuGomuNomi = gomuGomuNomi.drop(['G1', 'G2', 'G3'], axis=1)
gomuGomuNomi.head()

# In[3]:
# get_dummies = kolom palsu, sesuai dengan isi
# use one-hot encoding on categorical column
gomuGomuNomi = pd.get_dummies(gomuGomuNomi, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
gomuGomuNomi.head()

# In[4]:

# shuffle rows
gomuGomuNomi = gomuGomuNomi.sample(frac=1)
# split training and testing data
# Tampilkan Baris sampai dengan
gomuGomuNomi_baris = gomuGomuNomi[:500]
# Tampilkan Kolom sampai dengan
gomuGomuNomi_kolom = gomuGomuNomi[500:]

gomuGomuNomi_train_att = gomuGomuNomi_baris.drop(['pass'], axis=1)
gomuGomuNomi_train_pass = gomuGomuNomi_baris['pass']

gomuGomuNomi_test_att = gomuGomuNomi_kolom.drop(['pass'], axis=1)
gomuGomuNomi_test_pass = gomuGomuNomi_kolom['pass']

gomuGomuNomi_att = gomuGomuNomi.drop(['pass'], axis=1)
gomuGomuNomi_pass = gomuGomuNomi['pass']

# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(gomuGomuNomi_pass), len(gomuGomuNomi_pass), 100*float(np.sum(gomuGomuNomi_pass)) / len(gomuGomuNomi_pass)))

# In[5]:

# fit a decision tree
#t = semangka
from sklearn import tree

semangka = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
# fit menyesuaikan bobot nilai data sehingga akurasi lebih baik
semangka = semangka.fit(gomuGomuNomi_train_att, gomuGomuNomi_train_pass)


# In[6]:

# visualize tree
import graphviz
#t = naga
naga = tree.export_graphviz(semangka, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(gomuGomuNomi_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True)
#graph = pisang
pisang = graphviz.Source(naga)
pisang

# In[7]:

# save tree
tree.export_graphviz(semangka, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(gomuGomuNomi_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True)


# In[8]:

semangka.score(gomuGomuNomi_train_att, gomuGomuNomi_train_pass)

# In[9]:

from sklearn.model_selection import cross_val_score
# cross_val_score Evaluasi score berdasarkan validasi silang
# cv = pengecekan
scores = cross_val_score(semangka, gomuGomuNomi_att, gomuGomuNomi_pass, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# In[10]:

for max_depth in range(1, 20):
    semangka = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(semangka, gomuGomuNomi_att, gomuGomuNomi_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))

# In[11]:

apel_acc = np.empty((19,3), float)
print()()
i = 0
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(t, gomuGomuNomi_att, gomuGomuNomi_pass, cv=5)
    apel_acc[i,0] = max_depth
    apel_acc[i,1] = scores.mean()
    apel_acc[i,2] = scores.std() * 2
    i += 1
    
apel_acc

# In[12]:

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(apel_acc[:,0], apel_acc[:,1], yerr=apel_acc[:,2])
plt.show()


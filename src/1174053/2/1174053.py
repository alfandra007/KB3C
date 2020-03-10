# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:52:07 2020

@author: Sekar
"""
print(1174053%3)
#%% 1.Load Dataset
import pandas as pd
seblak = pd.read_csv('student-mat.csv',sep=';')
len(seblak)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
seblak['pass'] = seblak.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
seblak = seblak.drop(['G1','G2','G3'],axis=1)
seblak.head()

#%% 3.use one-hot encoding on categorical columns
seblak = pd.get_dummies(seblak,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
seblak.head()

#%% 4.shuffle rows
seblak = seblak.sample(frac=1)
seblak_train = seblak[:500]
seblak_test = seblak[500:]
seblak_train_att = seblak_train.drop(['pass'],axis=1)
seblak_train_pass = seblak_train['pass']
seblak_test_att = seblak_test.drop(['pass'],axis=1)
seblak_test_pass = seblak_test['pass']
seblak_att = seblak.drop(['pass'],axis=1)
seblak_pass = seblak['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(seblak_pass),len(seblak_pass),100*float(np.sum(seblak_pass))/len(seblak_pass)))
#%% 5.fit a decision tree
from sklearn import tree
surabi = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
surabi = surabi.fit(surabi_train_att,subang_train_pass)

#%% 6.visualize tree
import graphviz
martabak = tree.export_graphviz(surabi,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(seblak_train_att),class_names=["fail","pass"],filled=True,rounded=True)
baso = graphviz.Source(martabak)
baso

#%% 7.save tree
tree.export_graphviz(surabi,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(seblak_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
surabi.score(seblak_test_att,seblak_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
batagor = cross_val_score(surabi,seblak_att,seblak_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (batagor.mean(),batagor.std() * 2))

#%% 10
for siomay in range(1,20):
    surabi = tree.DecisionTreeClassifier(criterion="entropy",max_depth=siomay)
    batagor = cross_val_score(surabi,seblak_att,seblak_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(siomay,batagor.mean(),batagor.std() * 2))

#%% 11
pempek = np.empty((19,3),float)
tekwan = 0
for siomay in range(1,20):
    surabi = tree.DecisionTreeClassifier(criterion="entropy",max_depth=siomay)
    batagor = cross_val_score(surabi,seblak_att,seblak_pass,cv=5)
    pempek[tekwan,0] = siomay
    pempek[tekwan,1] = batagor.mean()
    pempek[tekwan,2] = batagor.std() * 2
    tekwan += 1
    pempek

#%% 12
import matplotlib.pyplot as plt
gudeg, basreng = plt.subplots()
basreng.errorbar(pempek[:,0],pempek[:,1],yerr=pempek[:,2])
plt.show()

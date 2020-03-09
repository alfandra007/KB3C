# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:04:58 2020

@author: Chandra
"""
# In[0]:
1174079 % 3
#Hasilnya = 2 maka mengambil variable dengan nama buah

# In[1]:
# load dataset (menggunakan student-mat)
import pandas as pd
apel = pd.read_csv('F:/Poltekpos/D4 TI 3C/Semester 6/Kecerdasan Buatan/Github/Upload 8 Maret 2020/src/1174079/2/dataset/student-mat.csv', sep=';')
len(apel)

# In[2]:
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
apel['pass'] = apel.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 30 else 0, axis=1)#mendeklarasikan pass/fail nya data berdasarkan G1+G2+G3.
apel = apel.drop(['G1','G2','G3'],axis=1)
apel.head() #


#%% 3.use one-hot encoding on categorical columns
apel= pd.get_dummies(apel,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
apel.head()#memanggil variabel dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data

#%% 4.shuffle rows
apel= apel.sample(frac=1)#mengembalikan variabel menjadi sampel acak dengan frac=1
apel_train = apel[:500]
apel_test = apel[500:]
apel_train_att = apel_train.drop(['pass'],axis=1)
apel_train_pass = apel_train['pass']
apel_test_att = apel_test.drop(['pass'],axis=1)
apel_test_pass = apel_test['pass']
apel_att = apel.drop(['pass'],axis=1)
apel_pass = apel['pass']

import numpy as np #mengimport module numpy sebagai np y 
print("Passing: %d out %d (%.2f%%)" %(np.sum(apel_pass),len(apel_pass),100*float(np.sum(apel_pass))/len(apel_pass)))
#%% 5.fit a decision tree
from sklearn import tree
mangga = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)  #membuat variabel bandung sebagai decisiontree, dengan criterion fungsi mengukur kualitas split
mangga = mangga.fit(apel_train_att,apel_train_pass)

#%% 6.visualize tree
import graphviz
pir = tree.export_graphviz(mangga,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(apel_train_att),class_names=["fail","pass"],filled=True,rounded=True)#mengambil data untuk diterjemahkan ke grafik
avokado = graphviz.Source(pir)
avokado

#%% 7.save tree
tree.export_graphviz(mangga,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(apel_train_att),class_names=["fail","pass"],filled=True,rounded=True)#save tree sebagai export graphviz ke file student-performance.dot

#%% 8
mangga.score(apel_att,apel_pass) #score juga disebut prediksi dengan diberi beberapa data input baru

#%% 9
from sklearn.model_selection import cross_val_score
anggur = cross_val_score(mangga,apel_att,apel_pass,cv=5) #mengevaluasi score dengan validasi silang
# show average score and +/- two standard deviations away 
print("Accuracy : %0.2f (+/- %0.2f)" % (anggur.mean(),anggur.std() * 2))

#%% 10
for belimbing in range(1,20):
    mangga = tree.DecisionTreeClassifier(criterion="entropy",max_depth=belimbing)
    anggur = cross_val_score(mangga,apel_att,apel_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(belimbing,anggur.mean(),anggur.std() * 2))
#Disini ini menunjukkan seberapa dalam di tree itu. Semakin dalam tree, semakin banyak perpecahan yang dimilikinya dan menangkap lebih banyak informasi tentang data.
#%% 11
stroberi = np.empty((19,3),float)
jeruk = 0   
for belimbing in range(1,20):
    mangga = tree.DecisionTreeClassifier(criterion="entropy",max_depth=belimbing)#variabel bandung untuk decision tree dengan ketentuan entropy
    anggur = cross_val_score(mangga,apel_att,apel_pass,cv=5)
    stroberi[jeruk,0] = belimbing
    stroberi[jeruk,1] = anggur.mean()
    stroberi[jeruk,2] = anggur.std() * 2
    jeruk += 1
    stroberi

#%% 12
import matplotlib.pyplot as plt
manggis, pisang = plt.subplots()
pisang.errorbar(stroberi[:,0],stroberi[:,1],yerr=stroberi[:,2]) #membuat error bar kemudian grafik akan ditampilkan menggunakan show
plt.show()
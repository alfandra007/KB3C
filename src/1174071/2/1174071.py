# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:11:23 2020

@author: muham
"""
print(1174071%3)
#%% 1.Load Dataset Student
import pandas as pd # load dataset (menggunakan student-mat.csv)
isekai = pd.read_csv('C:/Users/muham/Downloads/Compressed/KB3C-master_3/KB3C-master/src/1174071/2/dataset/student-mat.csv', sep=';') 
#variabel isekai memanggil fungsi untuk read file student-mat.csv
len(isekai) 
#mengetahui jumlah data baris pada data yang dipanggil

#%% 2.Men enerate binary label (pass/fail) based on G1+G2+G3
isekai['pass'] = isekai.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
#mendeklarasikan data pass/fail nya data berdasarkan G1+G2+G3.
isekai = isekai.drop(['G1','G2','G3'],axis=1)
#Mengetahui baris G1+G2+G3 ditambahkan, dan hasilnya sama dengan 35 maka axisnya 1.
isekai.head() 
#Memanggil variabel isekai untuk mengembalikan baris n atas 5 secara default dari frame atau seri data

#%% 3.Menggunakan one-hot encoding pada kolom kategori
isekai = pd.get_dummies(isekai,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
isekai.head()
#memanggil variabel isekai untuk mengembalikan baris n atas 5 secara default dari frame atau seri data

#%% 4.shuffle baris
isekai = isekai.sample(frac=1)
#Memamnggul fungsi sample acak dengan frac=1 pada variabel isekai 
isekai_train = isekai[:500]
isekai_test = isekai[500:]
isekai_train_att = isekai_train.drop(['pass'],axis=1)
isekai_train_pass = isekai_train['pass']
isekai_test_att = isekai_test.drop(['pass'],axis=1)
isekai_test_pass = isekai_test['pass']
isekai_att = isekai.drop(['pass'],axis=1)
isekai_pass = isekai['pass']

import numpy as np #Import modul numpy sebagai dengan alias np 
print("Passing: %d out %d (%.2f%%)" %(np.sum(isekai_pass),len(isekai_pass),100*float(np.sum(iskeai_pass))/len(isekai_pass)))
#%% 5.fit a decision tree
from sklearn import tree
raftel = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)  
#Membuat variabel raftelsebagai decisiontree, dengan criterion fungsi mengukur kualitas split
raftel = raftel.fit(isekai_train_att,isekai_train_pass)

#%% 6.visualize tree
import os
os.environ["PATH"] += os.pathsep + 'D:/graphviz-2.38/release/bin'

import graphviz
enies = tree.export_graphviz(raftel,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(isekai_train_att),class_names=["fail","pass"],filled=True,rounded=True)
#Mengubah data menjadi grafik
lobby = graphviz.Source(enies)
lobby

#%% 7.save tree
tree.export_graphviz(raftel,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(isekai_train_att),class_names=["fail","pass"],filled=True,rounded=True)
#Meng export data dari graphviz ke file student-performance.dot

#%% 8
raftel.score(isekai_test_att,isekai_test_pass) 
#Memprediksi dengan memberikan beberapa data baru

#%% 9
from sklearn.model_selection import cross_val_score
skypiea = cross_val_score(raftel,isekai_att,isekai_pass,cv=5) 
#Mengevaluasi score menggunakan validasi silang
print("Accuracy : %0.2f (+/- %0.2f)" % (skypiea.mean(),skypiea.std() * 2))

#%% 10
for water in range(1,20):
    raftel = tree.DecisionTreeClassifier(criterion="entropy",max_depth=water)
    skypiea = cross_val_score(raftek,isekai_att,isekai_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(arabasta,skypiea.mean(),skypiea.std() * 2))
#Menunjukkan data tree. Semakin dalam tree, semakin banyak perpecahan yang dimilikinya dan menangkap lebih banyak informasi tentang data.
#%% 11
seven = np.empty((19,3),float)
wano = 0
for water in range(1,20):
    raftel = tree.DecisionTreeClassifier(criterion="entropy",max_depth=water)
    #Membuatt variabel raftel untuk decision tree dengan ketentuan entropy
    skypiea = cross_val_score(raftel,isekai_att,isekai_pass,cv=5)
    seven[wano,0] = water
    seven[wano,1] = skypiea.mean()
    seven[wano,2] = skypiea.std() * 2
    wano += 1
    seven

#%% 12
import matplotlib.pyplot as plt
blitar, kediri = plt.subplots()
kediri.errorbar(seven[:,0],seven[:,1],yerr=seven[:,2]) 
#Membaut error pada bar kemudian grafik akan ditampilkan menggunakan show
plt.show()

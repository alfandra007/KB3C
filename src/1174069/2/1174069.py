# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:25:07 2020

@author: FannyShafira
"""

print(1174069%3)
#%% 1.Load Dataset
import pandas as pd # load dataset (menggunakan student-mat.csv)
padalarang = pd.read_csv('F://Semester 6/Artificial Intelligence/Tugas 2/src/dataset/student-mat.csv', sep=';') #variabel padalarang berfungsi untuk read file student-mat.csv
len(padalarang) #mengetahui jumlah baris pada data yang dipanggil

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
padalarang['pass'] = padalarang.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)#mendeklarasikan pass/fail nya data berdasarkan G1+G2+G3.
padalarang = padalarang.drop(['G1','G2','G3'],axis=1)#untuk mengetahui baris G1+G2+G3 ditambahkan, dan hasilnya sama dengan 35 maka axisnya 1.
padalarang.head() #memanggil variabel padalarang dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data

#%% 3.use one-hot encoding on categorical columns
padalarang = pd.get_dummies(padalarang,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
padalarang.head()#memanggil variabel padalarang dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data

#%% 4.shuffle rows
padalarang = padalarang.sample(frac=1)#mengembalikan variabel padalarang menjadi sampel acak dengan frac=1
padalarang_train = padalarang[:500]
padalarang_test = padalarang[500:]
padalarang_train_att = padalarang_train.drop(['pass'],axis=1)
padalarang_train_pass = padalarang_train['pass']
padalarang_test_att = padalarang_test.drop(['pass'],axis=1)
padalarang_test_pass = padalarang_test['pass']
padalarang_att = padalarang.drop(['pass'],axis=1)
padalarang_pass = padalarang['pass']

import numpy as np #mengimport module numpy sebagai np y 
print("Passing: %d out %d (%.2f%%)" %(np.sum(padalarang_pass),len(padalarang_pass),100*float(np.sum(padalarang_pass))/len(padalarang_pass)))
#%% 5.fit a decision tree
from sklearn import tree
bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)  #membuat variabel bandung sebagai decisiontree, dengan criterion fungsi mengukur kualitas split
bandung = bandung.fit(padalarang_train_att,padalarang_train_pass)

#%% 6.visualize tree
import os
os.environ["PATH"] += os.pathsep + 'D:/graphviz-2.38/release/bin'

import graphviz
bogor = tree.export_graphviz(bandung,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(padalarang_train_att),class_names=["fail","pass"],filled=True,rounded=True)#mengambil data untuk diterjemahkan ke grafik
jakarta = graphviz.Source(bogor)
jakarta

#%% 7.save tree
tree.export_graphviz(bandung,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(padalarang_train_att),class_names=["fail","pass"],filled=True,rounded=True)#save tree sebagai export graphviz ke file student-performance.dot

#%% 8
bandung.score(padalarang_test_att,padalarang_test_pass) #score juga disebut prediksi dengan diberi beberapa data input baru

#%% 9
from sklearn.model_selection import cross_val_score
depok = cross_val_score(bandung,padalarang_att,padalarang_pass,cv=5) #mengevaluasi score dengan validasi silang
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy : %0.2f (+/- %0.2f)" % (depok.mean(),depok.std() * 2))

#%% 10
for surabaya in range(1,20):
    bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=surabaya)
    depok = cross_val_score(bandung,padalarang_att,padalarang_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(surabaya,depok.mean(),depok.std() * 2))
#Disini ini menunjukkan seberapa dalam di tree itu. Semakin dalam tree, semakin banyak perpecahan yang dimilikinya dan menangkap lebih banyak informasi tentang data.
#%% 11
medan = np.empty((19,3),float)
sidoarjo = 0
for surabaya in range(1,20):
    bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=surabaya)#variabel bandung untuk decision tree dengan ketentuan entropy
    depok = cross_val_score(bandung,padalarang_att,padalarang_pass,cv=5)
    medan[sidoarjo,0] = surabaya
    medan[sidoarjo,1] = depok.mean()
    medan[sidoarjo,2] = depok.std() * 2
    sidoarjo += 1
    medan

#%% 12
import matplotlib.pyplot as plt
blitar, kediri = plt.subplots()
kediri.errorbar(medan[:,0],medan[:,1],yerr=medan[:,2]) #membuat error bar kemudian grafik akan ditampilkan menggunakan show
plt.show()
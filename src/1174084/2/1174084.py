# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:56:50 2020

@author: rezas
"""

# In[0]:

1174084 % 3
#Hasilnya = 1 maka mengambil variable dengan nama kota

# In[1]:
# load dataset (menggunakan student-mat)
import pandas as pd #Import library pandas menggantinya nama yang akan dipanggil jadi pd
mataram = pd.read_csv('D:/Git Kecerdasan Buatan/KB3C/src/1174084/2/dataset/student-mat.csv', sep=';') #Membuat variable tokyo yang isinya memanggil fungsi membaca file csv
len(mataram) #Menghitung jumlah data yang ada pada csv yang tadi sudah dibaca

# In[2]:
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
mataram['pass'] = mataram.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) #Membuat label binary (pass/fail) berdasarkan G1+G2+G3 (testgrade, semuanya 0-20 point); Batas untuk pass adalah sum>=30
mataram = mataram.drop(['G1', 'G2', 'G3'], axis=1) #Meghilangkan data G1 G2 dan G3
mataram.head() #Menampilkan data

# In[3]:
# use one-hot encoding on categorical columns
mataram = pd.get_dummies(mataram, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
mataram.head()

# In[4]:
# shuffle rows
mataram = mataram.sample(frac=1) #Mengambil data sample dari mataram
# split training and testing data
mataram_train = mataram[:500] #Membagi data untuk training
mataram_test = mataram[500:] #Membagi data untuk test

mataram_train_att = mataram_train.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
mataram_train_pass = mataram_train['pass'] #Mengambil data yang pass saja

mataram_test_att = mataram_test.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
mataram_test_pass = mataram_test['pass'] #Mengambil data yang pass saja

mataram_att = mataram.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
mataram_pass = mataram['pass'] #Mengambil data yang pass saja

# number of passing students in whole dataset:
import numpy as np #Mengimport library numpy sebagai np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(mataram_pass), len(mataram_pass), 100*float(np.sum(mataram_pass)) / len(mataram_pass))) #Menampilkan data


# In[5]:
# fit a decision tree
from sklearn import tree #import Decision tree dari library sklearn
bandung = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #Membuat decition tree dengan maximal depthnya 5 
bandung = bandung.fit(mataram_train_att, mataram_train_pass)#Memasukkan data yang akan dijadikan decition treenya

# In[6]:
# visualize tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/rezas/Anaconda3/Library/bin/graphviz/'
import graphviz #Mengimport Library Grapthviz untuk memvisualisasikan decision tree
malang = tree.export_graphviz(bandung, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(mataram_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True) #Mendefinisikan dot_data yang isikan akan berisikan data yang akan dijadikan gambar
jogja = graphviz.Source(malang) #Memasukkan data tadi menjadi sebuah jogja
jogja #Menampilkan jogja menggunakan graphviz

# In[7]:
# save tree
tree.export_graphviz(bandung, out_file="1174084.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(mataram_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True) #Digunakan untuk mengexport graph tree tadi yang telah kita buat

# In[8]:

bandung.score(mataram_test_att, mataram_test_pass) #Menghitung prediksi nilai yang akan datang dimasa depan

# In[9]:
from sklearn.model_selection import cross_val_score #Mengimport fungsi cross_val_score dari library sklearn
makassar = cross_val_score(bandung, mataram_att, mataram_pass, cv=5) #Mendefinisikan nagoya yang isinya pembagian data menjadi 5
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (makassar.mean(), makassar.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[10]:
for max_depth in range(1, 20): #Pengulangan menunjukkan seberapa dalam tree itu
    bandung = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #Membuat decision Tree
    makassar = cross_val_score(bandung, mataram_att, mataram_pass, cv=5) #Mendefinisikan nagoya yang isinya pembagian data menjadi 5
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, makassar.mean(), makassar.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi

# In[11]:
depth_acc = np.empty((19,3), float) #Membuat array baru
i = 0 #Membuat variable berisikan 0
for max_depth in range(1, 20): #Perulangan untuk memasukkan data 
    bandung = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #Membuat decision Tree
    makassar = cross_val_score(bandung, mataram_att, mataram_pass, cv=5) #Mendefinisikan nagoya yang isinya pembagian data menjadi 5
    depth_acc[i,0] = max_depth#Memasukkan data max_depth ke array depth_acc
    depth_acc[i,1] = makassar.mean() #Memasukkan data rata-rata dari nagoya ke array depth_acc
    depth_acc[i,2] = makassar.std() * 2 #Memasukkan data akar 2 dari nagoya ke array depth_acc
    i += 1
depth_acc


# In[12]:
import matplotlib.pyplot as plt #Menimport fungsi pyplot dari library matplotlib sebagai plt 

solo, denpasar = plt.subplots() #Membuat plot baru
denpasar.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])  #Mengisikan data plot
plt.show() #Menampilkan plot 

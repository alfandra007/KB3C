# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 22:51:53 2020

@author: Handi
"""

# In[0]
1174066 % 3 #Hasilnya 1 maka akan menggunakan nama Kota

# In[1]
import pandas as pd #Import library pandas menggantinya nama yang akan dipanggil jadi pd
tokyo = pd.read_csv('dataset/student-mat.csv', sep=';') #Membuat variable tokyo yang isinya memanggil fungsi membaca file csv
len(tokyo) #Menghitung jumlah data yang ada pada csv yang tadi sudah dibaca

# In[2]
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
tokyo['pass'] = tokyo.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) #Membuat label binary (pass/fail) berdasarkan G1+G2+G3 (testgrade, semuanya 0-20 point); Batas untuk pass adalah sum>=30
tokyo = tokyo.drop(['G1', 'G2', 'G3'], axis=1) #Meghilangkan data G1 G2 dan G3
tokyo.head() #Menampilkan data

# In[3]:
# use one-hot encoding on categorical columns
tokyo = pd.get_dummies(tokyo, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
tokyo.head()

# In[4]:
# shuffle rows
tokyo = tokyo.sample(frac=1) #Mengambil data sample dari tokyo
# split training and testing data
tokyo_train = tokyo[:500] #Membagi data untuk training
tokyo_test = tokyo[500:] #Membagi data untuk test

tokyo_train_att = tokyo_train.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
tokyo_train_pass = tokyo_train['pass'] #Mengambil data yang pass saja

tokyo_test_att = tokyo_test.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
tokyo_test_pass = tokyo_test['pass'] #Mengambil data yang pass saja

tokyo_att = tokyo.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
tokyo_pass = tokyo['pass'] #Mengambil data yang pass saja

# number of passing students in whole dataset:
import numpy as np #Mengimport library numpy sebagai np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(tokyo_pass), len(tokyo_pass), 100*float(np.sum(tokyo_pass)) / len(tokyo_pass))) #Menampilkan data

# In[5]:
# fit a decision tree
from sklearn import tree #import Decision tree dari library sklearn
kyoto = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #Membuat decition tree dengan maximal depthnya 5 
kyoto = kyoto.fit(tokyo_train_att, tokyo_train_pass) #Memasukkan data yang akan dijadikan decition treenya


# In[6]:
# visualize tree
import graphviz #Mengimport Library Grapthviz untuk memvisualisasikan decision tree
dot_data = tree.export_graphviz(kyoto, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(tokyo_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True) #Mendefinisikan dot_data yang isikan akan berisikan data yang akan dijadikan gambar
graph = graphviz.Source(dot_data) #Memasukkan data tadi menjadi sebuah graph
graph #Menampilkan graph menggunakan graphviz


# In[7]:
# save tree
tree.export_graphviz(kyoto, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(tokyo_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True) #Digunakan untuk mengexport graph tree tadi yang telah kita buat


# In[8]:
kyoto.score(tokyo_test_att, tokyo_test_pass) #Menghitung prediksi nilai yang akan datang dimasa depan


# In[9]:
from sklearn.model_selection import cross_val_score #Mengimport fungsi cross_val_score dari library sklearn
nagoya = cross_val_score(kyoto, tokyo_att, tokyo_pass, cv=5) #Mendefinisikan nagoya yang isinya pembagian data menjadi 5
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (nagoya.mean(), nagoya.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[10]:
for max_depth in range(1, 20): #Pengulangan menunjukkan seberapa dalam tree itu
    kyoto = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #Membuat decision Tree
    nagoya = cross_val_score(kyoto, tokyo_att, tokyo_pass, cv=5) #Mendefinisikan nagoya yang isinya pembagian data menjadi 5
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, nagoya.mean(), nagoya.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[11]:
depth_acc = np.empty((19,3), float) #Membuat array baru
i = 0 #Membuat variable berisikan 0
for max_depth in range(1, 20): #Perulangan untuk memasukkan data 
    kyoto = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)#Membuat decision Tree
    nagoya = cross_val_score(kyoto, tokyo_att, tokyo_pass, cv=5) #Mendefinisikan nagoya yang isinya pembagian data menjadi 5
    depth_acc[i,0] = max_depth #Memasukkan data max_depth ke array depth_acc
    depth_acc[i,1] = nagoya.mean() #Memasukkan data rata-rata dari nagoya ke array depth_acc
    depth_acc[i,2] = nagoya.std() * 2 #Memasukkan data akar 2 dari nagoya ke array depth_acc
    i += 1
    
depth_acc


# In[12]:
import matplotlib.pyplot as plt #Menimport fungsi pyplot dari library matplotlib sebagai plt 
fig, ax = plt.subplots() #Membuat plot baru
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) #Mengisikan data plot
plt.show() #Menampilkan plot 

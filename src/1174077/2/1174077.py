# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:45:04 2020

@author: Alvan
"""


# In[0]
1174077 % 3 #Hasilnya 0,menggunakan nama makanan

# In[1]
import pandas as pd #import library pandas dan sebagai pd
papeda = pd.read_csv('C:/Users/ASUS/Downloads/KB3C-master/KB3C-master/src/1174077/2/dataset/student-mat.csv', sep=';') #Membuat variable papeda yang isinya memanggil fungsi membaca file csv
len(papeda) #Menghitung jumlah data yang ada pada csv yang sudah dibaca

# In[2]
# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
papeda['pass'] = papeda.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) #Membuat label binary (pass/fail) berdasarkan G1+G2+G3 (testgrade, semuanya 0-20 point); Batas untuk pass adalah sum>=30
papeda = papeda.drop(['G1', 'G2', 'G3'], axis=1) #Meghilangkan data G1 G2 dan G3
papeda.head() #Menampilkan data

# In[3]:
# use one-hot encoding on categorical columns
papeda = pd.get_dummies(papeda, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])
papeda.head()

# In[4]:
# shuffle rows
papeda = papeda.sample(frac=1) #Mengambil data sample dari papeda
# split training and testing data
papeda_train = papeda[:500] #Membagi data untuk training
papeda_test = papeda[500:] #Membagi data untuk test

papeda_train_att = papeda_train.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
papeda_train_pass = papeda_train['pass'] #Mengambil data yang pass saja

papeda_test_att = papeda_test.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
papeda_test_pass = papeda_test['pass'] #Mengambil data yang pass saja

papeda_att = papeda.drop(['pass'], axis=1) #Meghapus data yang telah pass dan memasukkannya
papeda_pass = papeda['pass'] #Mengambil data yang pass saja

# number of passing students in whole dataset:
import numpy as np #Mengimport library numpy sebagai np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(papeda_pass), len(papeda_pass), 100*float(np.sum(papeda_pass)) / len(papeda_pass))) #Menampilkan data

# In[5]:
# fit a decision tree
from sklearn import tree #import Decision tree dari library sklearn
lontar = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #Membuat decition tree dengan maximal depthnya 5 
lontar = lontar.fit(papeda_train_att, papeda_train_pass) #Memasukkan data yang akan dijadikan decition treenya

# In[6]:
# visualize tree
import graphviz #Mengimport Library Graphviz untuk memvisualisasikan decision tree
dot_data = tree.export_graphviz(lontar, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(papeda_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True) #Mendefinisikan dot_data yang isikan akan berisikan data yang akan dijadikan gambar
graph = graphviz.Source(dot_data) #Memasukkan data tadi menjadi sebuah graph
graph #Menampilkan graph menggunakan graphviz

# In[7]:
# save tree
tree.export_graphviz(lontar, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(papeda_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True) #Digunakan untuk mengexport graph tree tadi yang telah kita buat


# In[8]:
lontar.score(papeda_test_att, papeda_test_pass) #Menghitung prediksi nilai yang akan datang dimasa depan


# In[9]:
from sklearn.model_selection import cross_val_score #Mengimport fungsi cross_val_score dari library sklearn
abon = cross_val_score(lontar, papeda_att, papeda_pass, cv=5) #Mendefinisikan abon yang isinya pembagian data menjadi 5
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (abon.mean(), abon.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[10]:
for max_depth in range(1, 20): #Pengulangan menunjukkan seberapa dalam tree itu
    lontar = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #Membuat decision Tree
    abon = cross_val_score(lontar, papeda_att, papeda_pass, cv=5) #Mendefinisikan abon yang isinya pembagian data menjadi 5
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, abon.mean(), abon.std() * 2)) #Menampilkan data nilai dan +/- dari dua standar deviasi


# In[11]:
depth_acc = np.empty((19,3), float) #Membuat array baru
i = 0 #Membuat variable berisikan 0
for max_depth in range(1, 20): #Perulangan untuk memasukkan data 
    lontar = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)#Membuat decision Tree
    abon = cross_val_score(lontar, papeda_att, papeda_pass, cv=5) #Mendefinisikan abon yang isinya pembagian data menjadi 5
    depth_acc[i,0] = max_depth #Memasukkan data max_depth ke array depth_acc
    depth_acc[i,1] = abon.mean() #Memasukkan data rata-rata dari abon ke array depth_acc
    depth_acc[i,2] = abon.std() * 2 #Memasukkan data akar 2 dari abon ke array depth_acc
    i += 1
    
depth_acc


# In[12]:
import matplotlib.pyplot as plt #Menimport fungsi pyplot dari library matplotlib sebagai plt 
fig, telur = plt.subplots() #Membuat plot baru
telur.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) #Mengisikan data plot
plt.show() #Menampilkan plot 
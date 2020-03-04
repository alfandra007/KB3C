# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

#%% Mencoba loading an example dataset
from sklearn import datasets
# Memanggil class datasets dari library sklearn
iris = datasets.load_iris() 
# Menggunakan contoh datasets iris
x = iris.data
# Menyimpan data sets iris pada variabel x
y = iris.target
# Menyimpan data label iris pada variabel y    

#%%Mencoba Learning dan predicting
from sklearn.neighbors import KNeighborsClassifier
#Memanggil fungsi KNeighborsClassifier dari library sklearn
import numpy as np 
#Memamnggil library numpy dengan alias np
knn=KNeighborsClassifier(n_neighbors=1) 
#Membuat variabel kkn dan memanggil fungsi KNeighborsClassifier lalu mendefinisikan k adalah 1
knn.fit(x,y)                            
#Perhitungan matematika kkn
a=np.array([1.0,2.0,3.0,4.0])           
#Membuat Array
a = a.reshape(1,-1)                     
#Mengubah Bentuk Array jadi 1 dimensi
hasil = knn.predict(a)                  
#Memanggil fungsi prediksi
print(hasil)                            
#menampilkan hasil prediksi

#%% Model Persistense
from sklearn import svm  
#Memangil class svm dari library sklearn
from sklearn import datasets 
#Memanggil class datasets dari library sklearn
clf = svm.SVC()              
#Membuat variabel dengan nama clf, dan memanggil class svm dan fungsi SVC
X, y = datasets.load_iris(return_X_y=True) 
#Mengambil dataset iris dan mengembalikan nilainya.
clf.fit(X, y)              
#Perhitungan nilai label

from joblib import dump, load 
#memanggil class dump dan load pada library joblib
dump(clf, '1174071.joblib') 
#Menyimpan model kedalam 1174071.joblib
hasil = load('1174071.joblib') 
#Memanggil model 1174071 dan disimpan pada variable hasil
print(hasil) 
#Menampilkan variable hasil

#%% Conventions
import numpy as np 
#memanggil library numpy dengan alias np
from sklearn import random_projection 
#Memanggil class random_projection dari library sklearn

rng = np.random.RandomState(0) 
#Membuat variabel rng, dan mendefisikan np, memanggil fungsi random dan attr RandomState kedalam variabel
X = rng.rand(10, 2000) 
#Membuat variabel X, dan menentukan nilai random dari 10 - 2000
X = np.array(X, dtype='float32') 
#Menyimpan hasil nilai random sebelumnya, kedalam array, dan menentukan typedatanya yaitu float32
X.dtype 
#Mengubah data tipe menjadi float64

transformer = random_projection.GaussianRandomProjection() 
#Membuat variabel transformer, dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
X_new = transformer.fit_transform(X) 
#Membuat variabel X_new dan melakukan perhitungan label pada variabel X
X_new.dtype 
#Mengubah data tipe menjadi float64
print(X_new)
#Menampilkan isi variabel X_new
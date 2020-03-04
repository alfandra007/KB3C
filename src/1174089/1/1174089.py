# -*- coding: utf-8 -*-
#%%Loading an example dataset
from sklearn import datasets # Load library dataset 
iris = datasets.load_iris() 
# variable iris diisi dengan contoh data
a = iris.data # Menyimpan value data ke variable A
b = iris.target # Menyimpan value data ke variable B

#%% Learning dan predicting
from sklearn.neighbors import KNeighborsClassifier
#Load library
import numpy as np
#load library

knn = KNeighborsClassifier(n_neighbors=1)
#mendefinisikan variabel bernama kkn, dan memanggil fungsi KNeighborsClassifier
# dan memberikan value 1
knn.fit(a,b) # perhitungan library knn

x = np.array([1.0,2.0,3.0,4.0]) 
# membuat array
x = x.reshape(1,-1)
#Convert array menjadi 1 dimensi

hasil = knn.predict(x)
#Memanggil fungsi predict dari KNN
print(hasil)                          
#menampilkan value dari variable hasil

#%% Model Persistense
from sklearn import svm  
# Load library
from sklearn import datasets 
# Load Library
clf = svm.SVC()              
# mendefinisikan variabel clf, dan memanggil fungsi SVC dari class svm
a, b = datasets.load_iris(return_X_y=True) 
#Variable a dan b diisi dengan dataset iris dan mengembalikan nilainya.
clf.fit(a, b)               
#memanggil fungsi fit dari clf

from joblib import dump, load
#Load library
dump(clf, '1174079.joblib') 
#Menyimpan model kedalam 1174079.joblib
hasil = load('1174079.joblib') 
#memuat model 1174079
print(hasil) # Menampilkan Hasil

#%% Conventions
import numpy as np 
# Load Library
from sklearn import random_projection 
#Load class random_projection dari library sklearn

rng = np.random.RandomState(0)
#Membuat variabel rng, dan mendefisikan np, fungsi random dan attr RandomState kedalam variabel
X = rng.rand(10, 2000) 
# membuat variabel X, dan menentukan nilai random dari 10 - 2000
X = np.array(X, dtype='float32') 
#menyimpan hasil nilai random sebelumnya, kedalam array, dan menentukan typedatanya sebagai float32
X.dtype 
# Mengubah data tipe menjadi float64

transformer = random_projection.GaussianRandomProjection() 
#membuat variabel transformer, dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
X_new = transformer.fit_transform(X) 
# membuat variabel baru dan melakukan perhitungan label pada variabel X
X_new.dtype 
# Mengubah data tipe menjadi float64
print(X_new) 
# Menampilkan isi variabel X_new




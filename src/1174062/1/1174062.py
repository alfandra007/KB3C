# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:39:40 2020

@author: USER
"""

#%% Mencoba loading an Example Dataset
from sklearn import datasets #digunakan untuk memanggil class dataset dari library sklearn
iris = datasets.load_iris()  # artinya kita menggunakan dataset iris
x = iris.data # artinya kita menyimpan data set iris di variable x
y = iris.target #artinya kita menyimpan data label iris pada variavle y

#%%Mmencoba Learning dan Predicting
from sklearn.neighbors import KNeighborsClassifier # kta menggunakan fungsi KNighborsClassfier pada kelas sklearn dan libarry sklearn
knn=KNeighborsClassifier(n_neighbors=1)  # kita membuta variable knn, dan memanggil function KNighbors
                                        
knn.fit(x,y) #kita membuat perhitungan matematika library knn                      
a=np.array([1.0,2.0,3.0,4.0])    # artinya kita membuat array       
a = a.reshape(1,-1)            # mengubah bentuk array jadi 1 dimensi        
hasil = knn.predict(a)   #kita memanggil fungsi prediksi               
print(hasil)  #menampilkan hasil prediksi

#%% Model Persistense
from sklearn import svm  #untuk memanggil class svm dari library sklearn
from sklearn import datasets #untuk class dataset dati library sklearn
clf = svm.SVC()    #kita membuat variable clf, dan memanggil class svm dan fungsi SVC          
X, y = datasets.load_iris(return_X_y=True) #Memanggil dataset iris dan mengembalikan nilainya
clf.fit(X, y)               # untuk menampilkan model yang di panggil sebelumnya

from joblib import dump, load #memanggil class dump dan load pada library joblib
dump(clf, '1174062.joblib') #menyimpan model kedalam 1174062.joblib
hasil = load('1174062.joblib') #memanggil model 1174062 
print(hasil)       #untuk menampilkan model yang dipanggil sebelumnya

#%% mencoba Conventions
import numpy as np #digunakan unutk memanggil library numpy dan dibuat alias np
from sklearn import random_projection #memanggil class random_projection pada library sklearn

rng = np.random.RandomState(0) #untuk membuat variable rng, mendefinisikan np, function random dan attr randomstate kedalam variable
X = rng.rand(10, 2000) #membuat variable X, dan menetukan nilai random dari 10-2000
X = np.array(X, dtype='float32') #untuk menyimpan hasil nilai random senelumnya, kedalam array dan menentukan typedatanya sebegai float32
X.dtype #untuk mengubah data tipe menjadi float64

transformer = random_projection.GaussianRandomProjection() #membuat varibale transformer dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
X_new = transformer.fit_transform(X) #untuk mebuat variable baru serta melakukan perhitungan label pada variabel X
X_new.dtype #mengubah data tipe menjadi float64
print(X_new) #menampilakn isi variable X_new
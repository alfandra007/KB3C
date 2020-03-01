# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:45:20 2020

@author: Bakti Qilan
"""

#%%Cara Dump Pertama
from sklearn import svm  #Mengimport sebuah Support Vector Machine(SVM) yang merupakan algoritma classification yang akan diambil dari Scikit-Learn.
from sklearn import datasets #Import fungsi datasets dari library sklearn
clf = svm.SVC() #Mendefinisikan clf dengan fungsi svc dari library svm
X, y = datasets.load_iris(return_X_y=True) #Mengisi variable x dan y dengan data dari datasets
clf.fit(X, y) #Estimator clf (for classifier)

import pickle #Mengimport Library pickle
s = pickle.dumps(clf) #Menyimpan hasil dari clf kedalam sebuah dump
clf2 = pickle.loads(s) #Memanggil dump yang dihasilkan pickle lalu memasukkan hasil dumpnya ke variable
clf2.predict(X[0:1]) #Memprediksi angka yang akan muncul
print(y[0]) #Menampilkan data prediksi

#%%Cara Dump Kedua
from sklearn import svm  # Digunakan untuk memangil class svm dari library sklearn
from sklearn import datasets # Diguankan untuk class datasets dari library sklearn
clf = svm.SVC()              # membuat variabel clf, dan memanggil class svm dan fungsi SVC
X, y = datasets.load_iris(return_X_y=True) #Mengambil dataset iris dan mengembalikan nilainya.
clf.fit(X, y)               #Perhitungan nilai label

from joblib import dump, load #memanggil class dump dan load pada library joblib
dump(clf, '1174083.joblib') #Menyimpan model kedalam 1174066.joblib
hasil = load('1174083.joblib') #Memanggil model 1174066
hasil.predict(X[0:1])
print(y[0]) # Menampilkan Model yang dipanggil sebelumnya


#%%Cara Dump Pertama


#Mengimport sebuah Support Vector Machine(SVM) yang merupakan algoritma classification yang akan diambil dari Scikit-Learn.
from sklearn import svm  
#Import fungsi datasets dari library sklearn
from sklearn import datasets 

#Mendefinisikan clf dengan fungsi svc dari library svm
clf = svm.SVC() 

#Mengisi variable x dan y dengan data dari datasets
X, y = datasets.load_iris(return_X_y=True) 

#Estimator clf (for classifier)
clf.fit(X, y) 

#Mengimport Library pickle
import pickle 

#Menyimpan hasil dari clf kedalam sebuah dump
s = pickle.dumps(clf) 

#Memanggil dump yang dihasilkan pickle lalu memasukkan hasil dumpnya ke variable
clf2 = pickle.loads(s) 


#Memprediksi angka yang akan muncul
clf2.predict(X[0:1]) 

#Menampilkan data prediksi
print(y[0]) 

#%%Cara Dump Kedua
# Digunakan untuk memangil class svm dari library sklearn
from sklearn import svm  

# Diguankan untuk class datasets dari library sklearn
from sklearn import datasets 

# membuat variabel clf, dan memanggil class svm dan fungsi SVC
clf = svm.SVC()              

#Mengambil dataset iris dan mengembalikan nilainya.
X, y = datasets.load_iris(return_X_y=True) 

#Perhitungan nilai label
clf.fit(X, y)               


#memanggil class dump dan load pada library joblib
from joblib import dump, load


#Menyimpan model kedalam 1174066.joblib
dump(clf, '1174083.joblib') 

#Memanggil model 1174066
hasil = load('1174083.joblib') 
hasil.predict(X[0:1])

# Menampilkan Model yang dipanggil sebelumnya

print(y[0]) 
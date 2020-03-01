# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:44:29 2020

@author: Bakti Qilan
"""

from sklearn import datasets #Import fungsi datasets dari library sklearn
iris = datasets.load_iris() #Memasukkan data dari datasets iris ke variable iris
digits = datasets.load_digits() #Memasukkan data dari datasets digits ke variable digits
from sklearn import svm #Mengimport sebuah Support Vector Machine(SVM) yang merupakan algoritma classification yang akan diambil dari Scikit-Learn.
clf = svm.SVC(gamma=0.001, C=100.) #Mendeklarasikan suatu value yang bernama clf yang berisi gamma.
clf.fit(digits.data[:-1], digits.target[:-1]) #Estimator clf (for classifier)
hasil = clf.predict(digits.data[-1:]) #Menunnjukkan prediksi angka baru
print(hasil)
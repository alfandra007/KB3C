# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:55:57 2020

@author: Rin
"""

import numpy as np # memanggil library numpy dan dibuat alias np
from sklearn import random_projection #Memanggil class random_projection pada library sklearn

rng = np.random.RandomState(0) #Membuat variabel rng, dan mendefisikan np, fungsi random dan attr RandomState kedalam variabel
X = rng.rand(10, 2000) # membuat variabel X, dan menentukan nilai random dari 10 - 2000
X = np.array(X, dtype='float32') #menyimpan hasil nilai random sebelumnya, kedalam array, dan menentukan typedatanya sebagai float32
X.dtype # Mengubah data tipe menjadi float64

transformer = random_projection.GaussianRandomProjection() #membuat variabel transformer, dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
X_new = transformer.fit_transform(X) # membuat variabel baru dan melakukan perhitungan label pada variabel X
X_new.dtype # Mengubah data tipe menjadi float64
print(X_new) # Menampilkan isi variabel X_new
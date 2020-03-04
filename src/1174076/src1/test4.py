
# memanggil library numpy dan dibuat alias np
import numpy as np 

#Memanggil class random_projection pada library sklearn
from sklearn import random_projection 

#Membuat variabel rng, dan mendefisikan np, fungsi random dan attr RandomState kedalam variabel
rng = np.random.RandomState(0) 

# membuat variabel X, dan menentukan nilai random dari 10 - 2000
X = rng.rand(10, 2000) 

#menyimpan hasil nilai random sebelumnya, kedalam array, dan menentukan typedatanya sebagai float32
X = np.array(X, dtype='float32') 

# Mengubah data tipe menjadi float64
X.dtype 

#membuat variabel transformer, dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
transformer = random_projection.GaussianRandomProjection() 
# membuat variabel baru dan melakukan perhitungan label pada variabel X
X_new = transformer.fit_transform(X) 
# Mengubah data tipe menjadi float64
X_new.dtype 
# Menampilkan isi variabel X_new
print(X_new) 

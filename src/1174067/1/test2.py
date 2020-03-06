
#Import fungsi datasets dari library sklearn
from sklearn import datasets

#Memasukkan data dari datasets iris ke variable iris
iris = datasets.load_iris() 

#Memasukkan data dari datasets digits ke variable digits
digits = datasets.load_digits() 

#Mengimport sebuah Support Vector Machine(SVM) yang merupakan algoritma classification yang akan diambil dari Scikit-Learn.
from sklearn import svm  

#Mendeklarasikan suatu value yang bernama clf yang berisi gamma.
clf = svm.SVC(gamma=0.001, C=100.) 

#Estimator clf (for classifier)
clf.fit(digits.data[:-1], digits.target[:-1]) 

#Menunnjukkan prediksi angka baru
hasil = clf.predict(digits.data[-1:]) 

#Menampilkan
print(hasil)
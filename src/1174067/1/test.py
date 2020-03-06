#Import fungsi datasets dari library sklearn
from sklearn import datasets 

#Memasukkan data dari datasets iris ke variable iris
iris = datasets.load_iris() 

#Memasukkan data dari datasets digits ke variable digits
digits = datasets.load_digits() 

#Menampilkan data dari datasets digits ke console
print(digits.data)
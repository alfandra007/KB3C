from sklearn import datasets #Import fungsi datasets dari library sklearn
iris = datasets.load_iris() #Memasukkan data dari datasets iris ke variable iris
digits = datasets.load_digits() #Memasukkan data dari datasets digits ke variable digits
print(digits.data) #Menampilkan data dari datasets digits ke console
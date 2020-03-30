# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:13:03 2020

@author: rezas
"""

# In[1]:
import pandas as pd #Mengimport pandas
d=pd.read_csv("D:/Git Kecerdasan Buatan/KB3C/src/1174084/4/Youtube02-KatyPerry.csv") #Membuat variable d untuk membaca file csv dari dataset
# In[2]:
spam=d.query('CLASS == 1') #mengelompokkan komentar spam
nospam=d.query('CLASS == 0') #mengelompokkan komentar bukan spam
# In[3]: memanggil lib vektorisasi
#melakukan fungsi bag of word dengan cara menghitung semua kata
#yang terdapat dalan file
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
# In[3]: 
dvec = vectorizer.fit_transform(d['CONTENT']) #melakukan bag of word pada dataframe pada colom CONTENT
# In[4]: 
dvec #melihat isi vektorisasi
# In[5]: 
print(d['CONTENT'][342]) #melihat isi data pada baris ke 342
# In[6]: 
daptarkata=vectorizer.get_feature_names() #feature_names merupakan digunakan untuk mengambil nama kolomnya ada apa saja
# In[7]:
dshuf = d.sample(frac=1) #melakukan randomisasi pada datanya supaya sempurna saat melakukan klasifikasi
# In[8]:
dk_train=dshuf[:300] #Data akan dibagi dari 300 row akhir menjadi data training dan sisanya adalah data testing
dk_test=dshuf[300:] #Data akan dibagi dari 300 row pertama menjadi data training dan sisanya adalah data testing
# In[9]: 
dk_train_att=vectorizer.fit_transform(dk_train['CONTENT']) #melakukan training pada data training dan di vektorisasi
print(dk_train_att)
# In[10]:
dk_test_att=vectorizer.transform(dk_test['CONTENT']) #melakukan testing pada data testing dan di vektorisasi
print(dk_test_att)
# In[11]:
dk_train_label=dk_train['CLASS']  #mengambil label spam dan bukan spam
print(dk_train_label)
dk_test_label=dk_test['CLASS'] #mengambil label spam dan bukan spam
print(dk_test_label)

# In[12]:
from sklearn import svm #Mengimport svm
clfsvm = svm.SVC() #clfsvm sebagai variabel untuk mengatur fungsi SVC
clfsvm.fit(dk_train_att, dk_train_label) #Mengatur data training
clfsvm.predict(dk_test_att)
clfsvm.score(dk_test_att, dk_test_label) #Mengatur data testing

# In[13]:
from sklearn import tree #Mengimport tree
clftree = tree.DecisionTreeClassifier() #clftree sebagai variabel untuk decision tree
clftree.fit(dk_train_att, dk_train_label) #Mengatur data training
# In[14]:
clftree.predict(dk_test_att) 
# In[15]:
clftree.score(dk_test_att, dk_test_label) #Mengatur data testing

# In[15]:
from sklearn.ensemble import RandomForestClassifier #Import fungsi randomforestclassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100) #clf sebagai variabel untuk klafisikasi random forest

clf.fit(dk_train_att, dk_train_label) #variable clf untuk fit yaitu menjadi data training
clf.score(dk_test_att, dk_test_label) #Memunculkan clf sebagai testing yang sudah di training 

# In[16]:
from sklearn.metrics import confusion_matrix #Mengimport Confusion Matrix
pred_labels = clf.predict(dk_test_att) #Membuat variable pred_labels dari data testing
cm = confusion_matrix(dk_test_label, pred_labels) #cm sebagai variabel data label
cm

# In[17]:
from sklearn.model_selection import cross_val_score #Mengimport cross_val_score
scores = cross_val_score(clf,dk_train_att,dk_train_label,cv=5) #Membuat variable scores sebagai variabel prediksi dari data training
scorerata2=scores.mean()
scorersd=scores.std()
# In[18]
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #menampilkan data scores dengan ketentuan akurasi

# In[19]:
import numpy as np
max_features_opts = range(1, 10, 1) #Variable max_features_opts sebagai variabel untuk membuat range 1, 10, 1
n_estimators_opts = range(2, 40, 4) #Variablen_estimators_opts sebagai variabel untuk membuat range 2, 40, 4
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4) , float) #Variable rf_params sebagai variabel untuk menjumlahkan yang sudah di tentukan sebelumnya
i = 0
for max_features in max_features_opts: #Perulangan 
    for n_estimators in n_estimators_opts: #Perulangan 
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators) #Menampilkan variabel csf
        scores = cross_val_score(clf, dk_train_att, dk_train_label, cv=5) #Variable scores sebagai variabel training 
        rf_params[i,0] = max_features #index 0
        rf_params[i,1] = n_estimators #index 1
        rf_params[i,2] = scores.mean() #index 2
        rf_params[i,3] = scores.std() * 2 #index 3
        i += 1 #Dengan ketentuan i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)"
% (max_features, n_estimators, scores.mean(), scores.std() * 2)) #Print hasil pengulangan yang sudah ditentukan
        
# In[20]:
import matplotlib.pyplot as plt #Mengimport library matplotlib sebagai plt
from mpl_toolkits.mplot3d import Axes3D #Mengimport axes3D untuk menampilkan plot 3 dimensi
from matplotlib import cm #Memanggil data cm yang sudah tersedia
fig = plt.figure() #Menghasilkan plot sebagai figure
fig.clf() #Figure di ambil dari clf
ax = fig.gca(projection='3d') #ax sebagai projection 3d
x = rf_params[:,0] #x sebagai index 0
y = rf_params[:,1] #y sebagai index 1
z = rf_params[:,2] #z sebagai index 2
ax.scatter(x, y, z) #Membuat plot scatter x y z
ax.set_zlim(0.6, 1) #Set zlim dengan ketentuan yang ada 
ax.set_xlabel('Max features') #Memberikan nama label x
ax.set_ylabel('Num estimators') #Memberikan nama label y
ax.set_zlabel('Avg accuracy') #Memberikan nama label z
plt.show() #Print hasil plot yang sudah dibuat.
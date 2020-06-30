# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:38:32 2020

@author: Rin
"""

# In[0]
import pandas as dirga # Melakukan import library pandas menjadi nama sendiri yaitu dirga

aplikasi = {"Nama Aplikasi" : ['VSCode','Atom','Sublime','Notepad++']} # Membuat varibel yang bernama aplikasi, dan mengisi dataframenya dengan nama nama aplikasi koding
x = dirga.DataFrame(aplikasi) # Membuat variabel x yang akan membuat DataFrame dari library pandas yang akan memanggil variabel aplikasi. 
print (' Dirga pake aplikasi: ' + x) #print hasil dari x

# In[1]
import numpy as dirga #Melakukan import library numpy menjadi nama sendiri yaitu dirga

matrix_x = dirga.eye(10) #Membuat sebuah matrix pake numpy dengan menggunakan fungsi eye
matrix_x #Mendeklrasikan matrix_x yang tadi dibuat

print (matrix_x) #print matrix_x yang tadi dibuat yang berbentuk 10x10

# In[2]
import matplotlib.pyplot as dirga #Melakukan import library numpy menjadi nama sendiri yaitu dirga
dirga.plot([1,1,7,4,0,6,6]) #Memasukkan nilai pada plot
dirga.xlabel('Dirga Brajamusti') #Menambahkan label pada x
dirga.ylabel('1174066') #Menambahkan label pada y
dirga.show() #Menampilkan grafik plot




# In[3]: Random Forest
import pandas as pd #Melakukan import library numpy menjadi pd

imgatt = pd.read_csv("N:/Tugas/Kuliah/Semester 6/Kecerdasan Buatan/KB3C Ngerjain/src/1174066/3/CUB_200_2011/attributes/image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0,1,2], names=['imgid', 'attid', 'present']) #Membuat variable imgatt untuk membaca file csv dari dataset, dengan ketentuan yang ada.

# In[4]

imgatt.head() #Menampilkan data yang sudah dibaca tadi tapi cuman data paling atas

# In[5]
imgatt.shape #Menampilkan jumlah seluruh data, kolom-nya

# In[6]
imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present') #Membuat sebuah variabel baru dari fungsi imgatt, dengan mengganti index menjadi kolom dan kolom menjadi index

# In[7]
imgatt2.head() #Menampilkan data yang sudah dibaca tadi tapi cuman data paling atas

# In[8]
imgatt2.shape #Menampilkan jumlah seluruh data, kolom-nya

imglabels = pd.read_csv("N:/Tugas/Kuliah/Semester 6/Kecerdasan Buatan/KB3C Ngerjain/src/1174066/3/CUB_200_2011/image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label']) #baca data csv dengan ketentuan yang ada

imglabels = imglabels.set_index('imgid') #variabel imglabels sebagai set index (imgid)

# In[9]
imglabels = pd.read_csv("N:/Tugas/Kuliah/Semester 6/Kecerdasan Buatan/KB3C Ngerjain/src/1174066/3/CUB_200_2011/image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label']) #Membaca data dimasukkan ke variable imglabels

imglabels = imglabels.set_index('imgid') #Variable imglabels dan set index (imgid)

# In[10]
imglabels.head() #Menampilkan data yang sudah dibaca tadi tapi cuman data paling atas

# In[11]
imglabels.shape #Menampilkan jumlah seluruh data, kolom-nya

# In[12]
df = imgatt2.join(imglabels) #Varibel df dimasukkan fungsi join dari data imgatt2 ke variabel imglabels
df = df.sample(frac=1) #Variabel df sebagai sample dengan ketentuan frac=1

# In[13]
df_att = df.iloc[:, :312] #Membuat kolom dengan ketentuan 312
df_label = df.iloc[:, 312:] #Membuat kolom dengan ketentuan 312

# In[14]
df_att.head() #Menampilkan data yang sudah dibaca tadi tapi cuman data paling atas

# In[15]
df_label.head() #Menampilkan data yang sudah dibaca tadi tapi cuman data paling atas

# In[16]
df_train_att = df_att[:8000] #Data akan dibagi dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_train_label = df_label[:8000] #Data akan dibagi dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_test_att = df_att[8000:] #Berbalik dari sebelumnya data akan dibagi mulai dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_test_label = df_label[8000:] #Berbalik dari sebelumnya data akan dibagi mulai dari 8000 row pertama menjadi data training dan sisanya adalah data testing

df_train_label = df_train_label['label'] #Menambahkan label
df_test_label = df_test_label['label'] #Menambahkan label

# In[17]
from sklearn.ensemble import RandomForestClassifier #Import fungsi randomforestclassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100) #clf sebagai variabel untuk klafisikasi random forest

# In[18]
clf.fit(df_train_att, df_train_label) #Variabnle clf untuk fit yaitu menjadi data training

# In[19]
print(clf.predict(df_train_att.head())) #Print clf yang di sudah prediksi dari training tetapi hanya menampilkan data paling atas

# In[20]
clf.score(df_test_att, df_test_label) #Memunculkan clf sebagai testing yang sudah di training tadi



# In[21]: Confusion Matrix 
from sklearn.metrics import confusion_matrix #Mengimport Confusion Matrix
pred_labels = clf.predict(df_test_att) #Membuat variable pred_labels dari data testing
cm = confusion_matrix(df_test_label, pred_labels) #cm sebagai variabel data label

# In[22]
cm #Memunculkan data label berbentuk array

# In[23]
import matplotlib.pyplot as plt #Mengimport library matplotlib sebagai plt
import itertools #Mengimport library itertools
def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): #Membuat fungsi dengan ketentuan data yang ada pada cm
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix") #Jika normalisasi sebagai ketentuan yang ada maka print normalized confusion matrix
    else:
        print('Confusion matrix, without normalization') #Jika tidak memenuhi kondisi if maka pring else

    print(cm) #Print data cm

    plt.imshow(cm, interpolation='nearest', cmap=cmap) #plt sebagai fungsi untuk membuat plot
    plt.title(title) #Membuat title pada plot 
    #plt.colorbar()
    tick_marks = np.arange(len(classes)) #Membuat marks pada plot
    plt.xticks(tick_marks, classes, rotation=90) #Membuat ticks pada x
    plt.yticks(tick_marks, classes) #Membuat ticks pada y

    fmt = '.2f' if normalize else 'd' #fmt sebagai normalisasi
    thresh = cm.max() / 2.  #Variable thresh menambil data max pada cm kemudian dibagi 2

    plt.tight_layout() #Mengatur layout pada plot
    plt.ylabel('True label') #Menambahkan nama label pada sumbu y
    plt.xlabel('Predicted label') #Menambahkan nama label pada sumbu x


# In[24]
birds = pd.read_csv("N:/Tugas/Kuliah/Semester 6/Kecerdasan Buatan/KB3C Ngerjain/src/1174066/3/CUB_200_2011/classes.txt",
                    sep='\s+', header=None, usecols=[1], names=['birdname']) #membaca csv dengan ketentuan nama birdname
birds = birds['birdname'] #nama birds dengan ketentuan birdname
birds #Menampilkan data birds


# In[25]
import numpy as np #Mengimport library numpy sebagai np
np.set_printoptions(precision=2) #np sebagai variabel yang membuat set precision=2
plt.figure(figsize=(60,60), dpi=300) #Plot sebagai figure dengan ketentuan sizw 60,60 dan dpi 300
plot_confusion_matrix(cm, classes=birds, normalize=True) #Data cm dan clas birds dibuat sebagai plot
#plt.show
#plt.savefig('hasil.png') #



# In[26]: Mencoba dengan metode Decission Tree dan SVM
from sklearn import tree #Mengimport library tree
clftree = tree.DecisionTreeClassifier() #clftree sebagai variabel untuk decision tree
clftree.fit(df_train_att, df_train_label) #Mengatur data training
clftree.score(df_test_att, df_test_label) #Mengatur data testing

# In[27]
from sklearn import svm #Mengimport library svm
clfsvm = svm.SVC() #clfsvm sebagai variabel untuk mengatur fungsi SVC
clfsvm.fit(df_train_att, df_train_label) #Mengatur data training
clfsvm.score(df_test_att, df_test_label) #Mengatur data testing

# In[28]: Pengecekan Cross Validation
from sklearn.model_selection import cross_val_score #Mengimport cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) #Membuat variable scores sebagai variabel prediksi dari data training
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #Print data scores dengan ketentuan akurasi

# In[29]
scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5) #Membuat variable prediksi menggunakan scores dan metode tree
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2)) #Menampilkan dengan ketentuan yang ada

# In[30]
scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5) #Membuat variable data training
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2)) #Menampilkan data testing dan output akurasi

# In[31]: Pengamatan komponen informasi
max_features_opts = range(5, 50, 5) #Variable max_features_opts sebagai variabel untuk membuat range 5,50,5
n_estimators_opts = range(10, 200, 20) #Variablen_estimators_opts sebagai variabel untuk membuat range 10,200,20
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float) #Variablerf_params sebagai variabel untuk menjumlahkan yang sudah di tentukan sebelumnya
i = 0
for max_features in max_features_opts: #Perulangan 
    for n_estimators in n_estimators_opts: #Perulangan
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators) #Menampilkan variabel csf
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) #Variable scores sebagai variabel training 
        rf_params[i,0] = max_features #index 0
        rf_params[i,1] = n_estimators #index 1
        rf_params[i,2] = scores.mean() #index 2
        rf_params[i,3] = scores.std() * 2 #index 3
        i += 1 #Dengan ketentuan i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %       (max_features, n_estimators, scores.mean(), scores.std() * 2))
        #Print hasil pengulangan yang sudah ditentukan

# In[32]
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
ax.set_zlim(0.2, 0.5) #Set zlim dengan ketentuan yang ada 
ax.set_xlabel('Max features') #Memberikan nama label x
ax.set_ylabel('Num estimators') #Memberikan nama label y
ax.set_zlabel('Avg accuracy') #Memberikan nama label z
plt.show() #Print hasil plot yang sudah dibuat.
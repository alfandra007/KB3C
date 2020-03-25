# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:48:08 2020

@author: rezas
"""
# In[1]:

import pandas as pd
nilai = {"mat":[70,80,75,90,90] , "ipa":[80,80,85,70,85] , "ips":[90,85,90,80,80]} 
df = pd.DataFrame(data=nilai) 
print("Original DataFrame")
print(df ) 
print("Baris Untuk nilai matematika dengan value 90")
print(df.loc[df ["mat"] == 90])

# In[2]:

import numpy as np
mat = np.array([70, 80, 75, 90, 90]) 
print ("Matematika : " , mat ) 
ipa = np.array ([80, 80, 85, 70, 85]) 
print ("IPA: " , ipa ) 
ips = np.array ([90, 85, 90, 80, 80]) 
print ("IPS: " , ips ) 
print ("Data Yang Sama Dari Nilai Matematika dan IPA Adalah :") 
print (np.intersect1d (mat , ipa))
print ("Data Yang Sama Dari Nilai Matematika dan IPS Adalah :") 
print (np.intersect1d (mat , ips))

# In[3]:

import matplotlib.pyplot as plt 
# line 1 points 
xmat = [70, 75, 80, 90] 
ymat = [1 ,1 ,1 ,2] 
# line 2 points 
xipa = [70, 80, 85] 
yipa = [1 ,2 ,2] 

xips = [80, 85, 90] 
yips = [2 ,1 ,2] 

# Set the x axis label of the current axis . 
plt.xlabel("x − Nilai") 
# Set the y axis label of the current axis . 
plt.ylabel("y − Jumlah") 
# Set a title 
plt.title("Grafik Nilai") 
# Display the figure . 
plt.plot(xmat ,ymat , color="salmon" , linewidth = 3, label = "Matematika") 
plt.plot (xipa ,yipa , color="mediumvioletred" , linewidth = 5, label = "IPA") 
plt.plot (xips ,yips , color="black" , linewidth = 7, label = "IPS")
# show a legend on the plot 
plt.legend() 
plt.show() 

# In[4]:

import pandas as pd #Import library numpy menjadi pd

imgatt = pd.read_csv("C:/Users/rezas/Downloads/CUB_200_2011/attributes/image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0,1,2], names=['imgid', 'attid', 'present']) #Membuat variable imgatt untuk membaca file csv dari dataset


imgatt.head() #Menampilkan data paling atas yang sudah dibaca 

imgatt.shape #Menampilkan jumlah seluruh data kolom
imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present') #Membuat sebuah variabel baru dari fungsi imgatt, dengan mengganti index menjadi kolom dan kolom menjadi index

imgatt2.head() #Menampilkan data paling atas yang sudah dibaca
imgatt2.shape #Menampilkan jumlah seluruh data kolom

imglabels = pd.read_csv("C:/Users/rezas/Downloads/CUB_200_2011/image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label']) #baca data csv  dan dimasukkan ke variable imglabels

imglabels = imglabels.set_index('imgid') #Variable imglabels dan set index (imgid)

imglabels.head() #Menampilkan data yang sudah dibaca tadi tapi cuman data paling atas

imglabels.shape #Menampilkan jumlah seluruh data, kolom-nya

df = imgatt2.join(imglabels) #Varibel df dimasukkan fungsi join dari data imgatt2 ke variabel imglabels
df = df.sample(frac=1) #Variabel df sebagai sample dengan ketentuan frac=1

df_att = df.iloc[:, :312] #Membuat kolom dengan ketentuan 312
df_label = df.iloc[:, 312:] #Membuat kolom dengan ketentuan 312

df_att.head() #Menampilkan data yang sudah dibaca hanya data paling atas

df_label.head() #Menampilkan data yang sudah dibaca hanya data paling atas

df_train_att = df_att[:8000] #Data akan dibagi dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_train_label = df_label[:8000] #Data akan dibagi dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_test_att = df_att[8000:] #data akan dibagi mulai dari 8000 row terakhir menjadi data training dan sisanya adalah data testing
df_test_label = df_label[8000:] # data akan dibagi mulai dari 8000 row terakhit menjadi data training dan sisanya adalah data testing

df_train_label = df_train_label['label'] #Menambahkan label
df_test_label = df_test_label['label'] #Menambahkan label

from sklearn.ensemble import RandomForestClassifier #Import fungsi randomforestclassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100) #clf sebagai variabel untuk klafisikasi random forest

clf.fit(df_train_att, df_train_label) #variable clf untuk fit yaitu menjadi data training
print(clf.predict(df_train_att.head())) #menampiulkan clf yang di sudah prediksi dari training yang data paling atas

clf.score(df_test_att, df_test_label) #Memunculkan clf sebagai testing yang sudah di training 

# In[5]:

from sklearn.metrics import confusion_matrix #Mengimport Confusion Matrix
pred_labels = clf.predict(df_test_att) #Membuat variable pred_labels dari data testing
cm = confusion_matrix(df_test_label, pred_labels) #cm sebagai variabel data label
cm
# In[6]:
from sklearn import tree #Mengimport tree
clftree = tree.DecisionTreeClassifier() #clftree sebagai variabel untuk decision tree
clftree.fit(df_train_att, df_train_label) #Mengatur data training
clftree.score(df_test_att, df_test_label) #Mengatur data testing

from sklearn import svm #Mengimport svm
clfsvm = svm.SVC() #clfsvm sebagai variabel untuk mengatur fungsi SVC
clfsvm.fit(df_train_att, df_train_label) #Mengatur data training
clfsvm.score(df_test_att, df_test_label) #Mengatur data testing

# In[7]:
from sklearn.model_selection import cross_val_score #Mengimport cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) #Membuat variable scores sebagai variabel prediksi dari data training
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #menampilkan data scores dengan ketentuan akurasi

# In[8]:

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
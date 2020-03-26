# In[1]: RANDOM FOREST

import pandas as pd #import library pandas sebagai as

imgatt = pd.read_csv("data/CUB_200_2011/attributes/image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0,1,2], names=['imgid', 'attid', 'present']) #library imgatt sebagai membaca file csv dari dataset, dengan ketentuan yang ada.


# In[2]:

imgatt.head() #menampilkan data yang di baca tadi tetapi hanya data paling atas


# In[3]:

imgatt.shape #menampilkan jumlah data, kolom


# In[4]:

imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present') #membuat variabel baru dari fungsi imgatt, dengan mengganti index dan kolom (kebalikan)


# In[5]:

imgatt2.head() #menampilkan data yang di baca tadi tetapi hanya data paling atas


# In[6]:

imgatt2.shape #menampilkan jumlah data, kolom


# In[7]:

imglabels = pd.read_csv("data/CUB_200_2011/image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label']) #baca data csv dengan ketentuan yang ada

imglabels = imglabels.set_index('imgid') #variabel imglabels sebagai set index (imgid)


# In[8]:

imglabels.head() #menampilkan data yang di baca tadi tetapi hanya data paling atas


# In[9]:

imglabels.shape #menampilkan jumlah data, kolom


# In[10]:

df = imgatt2.join(imglabels) #varibel df sebagai fungsi join dari data imgatt2 ke variabel imglabels
df = df.sample(frac=1) #variabel df sebagai sample dengan ketentuan frac=1


# In[11]:
    
df_att = df.iloc[:, :312] #membuat kolom dengan ketentuan 312
df_label = df.iloc[:, 312:] #membuat kolom dengan ketentuan 312


# In[12]:

df_att.head() #menampilkan data yang di baca tadi tetapi hanya data paling atas


# In[13]:

df_label.head() #menampilkan data yang di baca tadi tetapi hanya data paling atas


# In[14]:

df_train_att = df_att[:8000] #akan membagi 8000 row pertama menjadi data training dan sisanya adalah data testing
df_train_label = df_label[:8000] #akan membagi 8000 row pertama menjadi data training dan sisanya adalah data testing
df_test_att = df_att[8000:] #kebalikan dari akan membagi 8000 row pertama menjadi data training dan sisanya adalah data testing
df_test_label = df_label[8000:] #kebalikan dari akan membagi 8000 row pertama menjadi data training dan sisanya adalah data testing

df_train_label = df_train_label['label'] #menampilkan data training 
df_test_label = df_test_label['label'] #menampilkan data testing


# In[15]:

from sklearn.ensemble import RandomForestClassifier #import randomforestclassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100) #clf sebagai variabel untuk klafisikasi random forest


# In[16]:

clf.fit(df_train_att, df_train_label) #variabel clf untuk fit yaitu training


# In[17]:

print(clf.predict(df_train_att.head())) #print clf yang di prediksi dari training tetapi hanya menampilkan data paling atas


# In[18]:

clf.score(df_test_att, df_test_label) #print clf sebagai testing yang sudah di training tadi


# In[19]: Confusion Matrix 

from sklearn.metrics import confusion_matrix #import Confusion Matrix
pred_labels = clf.predict(df_test_att) #sebagai data testing
cm = confusion_matrix(df_test_label, pred_labels) #cm sebagai variabel data label


# In[20]:

cm #menampilkan data label berbentuk array


# In[21]:

import matplotlib.pyplot as plt #import library matplotlib sebagai plt
import itertools #import library itertools
def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): #membuat fungsi dengan ketentuan data yang ada pada cm
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix") #jika normalisasi sebagai ketentuan yang ada maka print normalized confusion matrix
    else:
        print('Confusion matrix, without normalization') #jika tidak memenuhi kondisi if maka pring else

    print(cm) #print data cm

    plt.imshow(cm, interpolation='nearest', cmap=cmap) #plt sebagai fungsi untuk membuat plot
    plt.title(title) #membuat title pada plot 
    #plt.colorbar()
    tick_marks = np.arange(len(classes)) #membuat marks pada plot
    plt.xticks(tick_marks, classes, rotation=90) #membuat ticks pada x
    plt.yticks(tick_marks, classes) #membuat ticks pada y

    fmt = '.2f' if normalize else 'd' #fmt sebagai normalisasi
    thresh = cm.max() / 2.  #variabel thresh menambil data max pada cm kemudian dibagi 2

    plt.tight_layout() #mengatur layout pada plot
    plt.ylabel('True label') #memberi nama label pada sumbu y
    plt.xlabel('Predicted label') #memberi nama label pada sumbu x


# In[22]:

birds = pd.read_csv("data/CUB_200_2011/classes.txt",
                    sep='\s+', header=None, usecols=[1], names=['birdname']) #membaca csv dengan ketentuan nama birdname
birds = birds['birdname'] #nama birds dengan ketentuan birdname
birds #menampilkan data birds


# In[23]:

import numpy as np #import library numpy sebagai np
np.set_printoptions(precision=2) #np sebagai variabel yang membuat set precision=2
plt.figure(figsize=(60,60), dpi=300) #plot sebagai figure dengan ketentuan sizw 60,60 dan dpi 300
plot_confusion_matrix(cm, classes=birds, normalize=True) #data cm dan clas birds dibuat sebagai plot
plt.show() #menampilkan hasil plot yang berbentuk grafik


# In[24]: Mencoba dengan metode Decission Tree dan SVM

from sklearn import tree #import library tree
clftree = tree.DecisionTreeClassifier() #clftree sebagai variabel untuk decision tree
clftree.fit(df_train_att, df_train_label) #sebagai data training
clftree.score(df_test_att, df_test_label) #sebagai data testing


# In[25]:

from sklearn import svm #import library svm
clfsvm = svm.SVC() #clfsvm sebagai variabel untuk mengatur fungsi SVC
clfsvm.fit(df_train_att, df_train_label) #sebagai data training
clfsvm.score(df_test_att, df_test_label) #sebagai data testing


# In[26]: Pengecekan Cross Validation

from sklearn.model_selection import cross_val_score #import cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) #variabel scores sebagai variabel prediksi dari data training
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #print data scores dengan ketentuan akurasi


# In[27]:

scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5) #sebagai prediksi menggunakan scores dan metode tree
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2)) #menampilkan dengan ketentuan yang ada


# In[28]:

scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5) #sebagai data training
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2)) #sebagai data testing dan output akurasi


# In[29]: Pengamatan komponen informasi

max_features_opts = range(5, 50, 5) #max_features_opts sebagai variabel untuk membuat range 5,50,5
n_estimators_opts = range(10, 200, 20) #n_estimators_opts sebagai variabel untuk membuat range 10,200,20
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float) #rf_params sebagai variabel untuk menjumlahkan yang sudah di tentukan sebelumnya
i = 0
for max_features in max_features_opts: #pengulangan 
    for n_estimators in n_estimators_opts: #pengulangan
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators) #menampilkan variabel csf
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) #scores sebagai variabel training 
        rf_params[i,0] = max_features #index 0
        rf_params[i,1] = n_estimators #index 1
        rf_params[i,2] = scores.mean() #index 2
        rf_params[i,3] = scores.std() * 2 #index 3
        i += 1 #dengan ketentuan i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %       (max_features, n_estimators, scores.mean(), scores.std() * 2))
        #print hasil pengulangan yang sudah ditentukan

# In[30]:

import matplotlib.pyplot as plt #import library matplotlib sebagai plt
from mpl_toolkits.mplot3d import Axes3D #import axes3D untuk menampilkan plot 3 dimensi
from matplotlib import cm #memanggil data cm yang sudah tersedia
fig = plt.figure() #hasil plot sebagai figure
fig.clf() #figure di ambil dari clf
ax = fig.gca(projection='3d') #ax sebagai projection 3d
x = rf_params[:,0] #x sebagai index 0
y = rf_params[:,1] #y sebagai index 1
z = rf_params[:,2] #z sebagai index 2
ax.scatter(x, y, z) #membuat plot scatter x y z
ax.set_zlim(0.2, 0.5) #set zlim dengan ketentuan yang ada 
ax.set_xlabel('Max features') #memberi nama label x
ax.set_ylabel('Num estimators') #memberi nama label y
ax.set_zlabel('Avg accuracy') #memberi nama label z
plt.show() #print hasil plot yang sudah dibuat.


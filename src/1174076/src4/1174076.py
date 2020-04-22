# Soal 1
# Mengimport library pandas
import pandas as pd

# Membuat variabel untuk membaca file
d = pd.read_csv('MOCK_DATA.csv', sep=';', nrows=500)
# Memasukkan data kedalam dataframe
df = pd.DataFrame(d, columns=['id','first_name','last_name','email',
                              'gender','matematika','fisika'])

dummy = pd.get_dummies(df['first_name'])
dummy.head()
df = df.join(dummy)


# In[]
# Soal 2
# variabel yang menampung Data sebanyak 450
allData = d[:450] 

# Variabel yang menampung data sebanyak 50 atau data sisa
restData = d[450:]

# In[]
# Soal 3
npm = 1174076 % 4
print(npm)

# Mengimport library pandas
import pandas as pd

# Membuat variabel untuk membaca file
d = pd.read_csv('Youtube02-KatyPerry.csv')

from sklearn.feature_extraction.text import CountVectorizer

# Variabel yang menghiung jumlah kata yang muncul pada setiap dokumen
vectorizer = CountVectorizer()

# Variabel yang akan di traning
# fit_transform kombinasi dari 2 function
# Fit untuk menghasilkan paramter model pembelajaran dari data data train
# Transform parameter yang dihasilkan oleh fit(), menerapkan model untuk menghasilkan transform dataset
dvec = vectorizer.fit_transform(d['CONTENT'])

# variabel yang menjalankan function yang mendapatkan semua kata pada dokumen
daptarkata = vectorizer.get_feature_names()

# variabel yang menampung seluruh data lalu dikocok agar random
dshuf = d.sample(frac=1)

# variabel yang menampung Data sebanyak 300
d_train = dshuf[:300] 

# varibal yang menampung data sisa
d_test = dshuf[300:]

# lakukan training
d_train_att = vectorizer.fit_transform(d_train['CONTENT'])
d_train_att

# lakukan transform
d_test_att = vectorizer.transform(d_test['CONTENT'])
d_test_att

# Variabel yang menampung data pada kolom CLASSs
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']

# In[]
# Soal 4
# import library
from sklearn import svm

# Variabel yang menampung library 
clfsvm = svm.SVC()

# Memprediksi data dari data training
clfsvm.fit(d_train_att, d_train_label)

# Varibael yang menampung hasil data yang sudah di training
clfsvm.score(d_test_att, d_test_label)

# In[]
# Soal 5
# Mengimport tree dari sklearn
from sklearn import tree 

# Variabel yang menampung library 
clftree = tree.DecisionTreeClassifier()

# Memprediksi data dari data training
clftree.fit(d_train_att, d_train_label)

# Varibael yang menampung hasil data yang sudah di training
clftree.score(d_test_att, d_test_label) 


# In[]
#Soal 6
from sklearn.metrics import confusion_matrix
pred_labelstree = clftree.predict(d_test_att)
cmtree = confusion_matrix(d_test_label, pred_labelstree)
cmtree

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

types = pd.read_csv("classes.txt",sep='\s+', header=None,usecols=[1], names=['type'])
types = types['type']
types

import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4), dpi=100)
plot_confusion_matrix(cmtree, classes=types, normalize=True)
plt.show()

# In[]
#Soal 7
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)
clf.fit(d_train_att, d_train_label)
clf.score(d_test_att, d_test_label)

from sklearn.model_selection import cross_val_score

scorestree = cross_val_score(clftree, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

scoressvm = cross_val_score(clfsvm, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[]
#Soal 8
max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" % (max_features, n_estimators, scores.mean(), scores.std() * 2))

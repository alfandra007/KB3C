# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:33:29 2020

@author: rezas
"""

# In[1]:import lib
import csv
#Mengimport library csv untuk mengimport file ekstensi .csv
from PIL import Image as pil_image
#Mengimport library Image dari PIL sebagai pil_image yang digunakan untuk mengolah data gambar 
import keras.preprocessing.image
#Mengimport library keras dengan metode preprocessing.image yang digunakan untuk membuat neural network

# In[2]:load all images (as numpy arrays) and save their classes

imgs = []
#Membuat variabel imgs dengan variabel kosong
classes = []
#Membuat variabel imgs dengan variabel kosong
with open('D:/New folder/KB3C/src/1174084/7/HASYv2/hasy-data-labels.csv') as csvfile:
    #membuka file csv pada folder HASYv2 yaitu hasy-data-labels.csv sebagai csvfile
    csvreader = csv.reader(csvfile)
    #membuat variabel csvreader yang berisikan metode reader dari library csv yang membaca csvfile.
    i = 0
    #Membuat varianel i yang berisikan 0
    for row in csvreader:
    #Membuat pengulangan pada variabel csvreader
        if i > 0:
        #Dengan ketentuan jika i lebih besar dari 0
            img = keras.preprocessing.image.img_to_array(pil_image.open("D:/New folder/KB3C/src/1174084/7/HASYv2/" + row[0]))
            #Membuat variabel img yang berisikan fungsi keras untuk aktivasi neural network yang membaca data pada folder HASYv2 yang dibuka dengan row berparameter 0.
            # neuron activation functions behave best when input values are between 0.0 and 1.0 (or -1.0 and 1.0),
            # so we rescale each pixel value to be in the range 0.0 to 1.0 instead of 0-255
            img /= 255.0
            #Membagi data yang berada pada variabel img dengan 255.0
            imgs.append((row[0], row[2], img))
            #Menambahkan nilai baru pada imgs yaitu row 0,row 2 dilanjutkan dengan variabel img.
            classes.append(row[2])
            # menambahkan nilai pada row ke 2 pada variabel classes
        i += 1
        #Menambahkan nilai 1 pada variabel i

# In[3]:shuffle the data, split into 80% train, 20% test

import random
#Mengimport library random
random.shuffle(imgs)
#Melakukan shuffle pada variabel imgs
split_idx = int(0.8*len(imgs))
#Membuat variabel split_idx yang diisi dengan nilai integer dari perkaslian 80% dengan jumlah dari variabel imgs
train = imgs[:split_idx]
#Membuat variabel train yang diisi dengan dengan pemecahan index awal pada data variabel split_idx 
test = imgs[split_idx:]
#Membuat variabel test yang diisi dengan pemecahan index akhir pada data variabel split_idx

# In[4]: 

import numpy as np
#Mengimport library numpy sebagai np
train_input = np.asarray(list(map(lambda row: row[2], train)))
#Membuat variabel train_input dengan np method asarray yang mana membuat array dangan fungsi list yang didalamnya diterapkan fungsi map untuk mengembalikan interator
#dan menggunakan lamba untuk mengecilkan fungsi dari objek yang berada pada row 2 dari data train
test_input = np.asarray(list(map(lambda row: row[2], test)))
#Membuat variabel test_input dengan isi np method asarray yang mana membuat array dangan fungsi list yang didalamnya diterapkan fungsi map untuk mengembalikan interator
#dan menggunakan lamba untuk mengecilkan fungsi dari objek yang berada pada row 2 dari data test
train_output = np.asarray(list(map(lambda row: row[1], train)))
#Membuat variabel train_output dengan np method asarray yang mana membuat array dangan fungsi list yang didalamnya diterapkan fungsi map untuk mengembalikan interator
#dan menggunakan lamba untuk mengecilkan fungsi dari objek yang berada pada row 1 dari data train
test_output = np.asarray(list(map(lambda row: row[1], test)))
#Membuat variabel test_output dengan np method asarray yang mana membuat array dangan fungsi list yang didalamnya diterapkan fungsi map untuk mengembalikan interator
#dan menggunakan lamba untuk mengecilkan fungsi dari objek yang berada pada row 1 dari data train

# In[5]: import encoder and one hot
from sklearn.preprocessing import LabelEncoder
#Mengimport library LabelEncoder dari sklearn.preprocessing yang digunakan untuk mengonversi jenis data teks kategori menjadi data numerik
from sklearn.preprocessing import OneHotEncoder
#Mengimport library OneHotEncoder dari sklearn.preprocessing

# In[6]:convert class names into one-hot encoding

# first, convert class names into integers
label_encoder = LabelEncoder()
#Membuat variabel label_encoder dengan isi LabelEncoder
integer_encoded = label_encoder.fit_transform(classes)
#Membuat variabel integer_encoded yang berfungsi untuk mengkonversi variabel classes kedalam bentuk integer

# In[7]:then convert integers into one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
#Membuat variabel onehot_encoder dengan isi fungsi OneHotEncoder parameter sparse=false
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#Membuat variabel integer_encoder dengan isi fungsi integer_encoded yang telah di convert pada fungsi sebelumnya
onehot_encoder.fit(integer_encoded)
#Onehotencoding melakukan fitting pada variabel integer_encoded

# In[8]:convert train and test output to one-hot
train_output_int = label_encoder.transform(train_output)
#Membuat variabel train_output_int dengan isi hasil konversi data train_output menggunakan label_encoder
train_output = onehot_encoder.transform(train_output_int.reshape(len(train_output_int), 1))
#Mengkonversi data train_output_int menggunakan fungsi onehot_encoder
test_output_int = label_encoder.transform(test_output)
#Membuat variabel test_output_int dengan isi hasil konversi data test_output menggunakan label_encoder
test_output = onehot_encoder.transform(test_output_int.reshape(len(test_output_int), 1))
#Mengkonversi data test_output_int menggunakan fungsi onehot_encoder
num_classes = len(label_encoder.classes_)
#Membuat variabel num_classes dengan isi jumlah class pada label_encoder
print("Number of classes: %d" % num_classes)
#Menampilkan hasil dari variabel num_classes

# In[9]: import sequential
import tensorflow as tf
from keras.models import Sequential
#Mengimport Sequential dari library keras
from keras.layers import Dense, Dropout, Flatten
#Mengimport Dense, Dropout, Flatten dari Library keras
from keras.layers import Conv2D, MaxPooling2D
#Mengimport Conv2D dan MaxPoolinf2D dari library Keras

# In[10]: desain jaringan
model = tf.keras.Sequential()
#Membuat variabel model dengan isi fungsi Sequential
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=np.shape(train_input[0])))
#Variabel model ditambahkan fungsi Conv2d dengan paramater 32 filter dengan karnel berukuran 3x3 
#dengan algoritam activation relu mengunakan data train_input mulai dari baris nol.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#Variabel madel ditambahkan fungsi MaxPooling2D dengan ketentuan ukuran 2x2
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#Variabel model ditambahkan fungsi Conv2D dengan 32 filter dengan konvolusi berukuran 3x3, menggunakan algoritam activation relu
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#Variabel madel ditambahkan fungsi MaxPooling2D dengan ketentuan ukuran 2x2
model.add(tf.keras.layers.Flatten())
#Variabel model di tambahkan fungsi Flatten
model.add(tf.keras.layers.Dense(1024, activation='tanh'))
#Variabel model ditambahakan fungsi dense dengan 1024 neuron, dan menggunakan algoritma tanh untuk activation
model.add(tf.keras.layers.Dropout(0.5))
#Variabel model di tambahkan fungsi Dropout sebesar 50% untuk mencegah terjadinya overfitting
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
#Variabel model ditambahkan fungsi Dense dengan parameter variabel num_classes menggunakan activation softmax
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#Variabel model di compile dengan parameter loss, matrik, dan optimasi
print(model.summary())
#Menampilkan model yang telah dibuat.

# In[11]: import sequential

import keras.callbacks
#mengimport library keras callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir='.\\logs\\mnist-style')
#Membuat variabel tensorboard dengan isi fungsi TenserBoard yang ada pada library keras.callback dengan parameter director log './logs/mnist-style'

# In[12]: 5menit kali 10 epoch = 50 menit
model.fit(train_input, train_output,
#Melakaukan fitting pada model dengan paramater train_input,train_output
          batch_size=32,
          #dengan mengunakan batch_size sebesar 32,
          epochs=10,
          #epoche=10 yang berarti terajadi perulanagan sebanyak 10 kali
          verbose=2,
          #untuk menghasilkan informasi logging dari data yang ditentukan dengan nilai 2
          validation_split=0.2,
          #melakukan pemecahan nilai sebesar 0.2 / 20% dari perhitungan validasi
          callbacks=[tensorboard])
          #mengeksekusi tensorboard dimana digunakan untuk visualisasikan parameter training, metrik, hiperparameter pada nilai/data yang diproses

score = model.evaluate(test_input, test_output, verbose=2)
#Membuat variabel score dengan isi fungsi evuleate dari model dengan paramater test_input, test_output dan verbose=2
#untuk memprediksi output dan input
print('Test loss:', score[0])
#Menampilkan score optimasi dengan ketentuan nilai parameter 0
print('Test accuracy:', score[1])
#Mencetak score akurasi dengan ketentuan nilai parameter 1

# In[13]:try various model configurations and parameters to find the best

import time
#import library time

results = []
#Membuat variabel result dengan isi array kosong
for conv2d_count in [1, 2]:
#Melakukan looping mengunakan convd2d_count dengan ketentuan konvolusi 2 dimensi yaitu 1, 2
    for dense_size in [128, 256, 512, 1024, 2048]:
    #Melakukan looping menggunakan ukuran dari densenya yaitu 123, 256, 512, 1024, 2048.
        for dropout in [0.0, 0.25, 0.50, 0.75]:
        #Melakukan looping menggunakn dropout dengan ketentuan 0%, 25%, 50%, 75% untuk memangkas data.
            model = tf.keras.Sequential()
            #Membuat variabel model dengan isi fungsi sequential.
            for i in range(conv2d_count):
            #Melakukan looping menggunakan i dengan jarak hasil konvulasi
                if i == 0:
                #jika nilai i sama dengan 0
                    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
                    #Variabel model ditambahkan fungsi Conv2d dengan paramater 32 filter dengan karnel berukuran 3x3 
                    #dengan algoritam activation relu mengunakan data train_input mulai dari baris nol.
                else:
                #jika tidak
                    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
                    #Variabel model akan ditambahkan fungsi Conv2d dengan paramater 32 filter dengan karnel berukuran 3x3
                    #dengan algoritam activation relu tanpa ada parameter input_shape.
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
                #Variabel madel ditambahkan fungsi MaxPooling2D dengan ketentuan ukuran 2x2
            model.add(tf.keras.layers.Flatten())
            #Variabel model di tambahkan fungsi Flatten
            model.add(tf.keras.layers.Dense(dense_size, activation='tanh'))
            #Variabel model ditambahakan fungsi dense dengan jumlah dense yang digunakan, dan menggunakan algoritma tanh untuk activation
            if dropout > 0.0:
            #jika nilai dari dropout lebih besar dari 0.0
                model.add(tf.keras.layers.Dropout(dropout))
                #Variabel model di tambahkan fungsi Dropout sebesar dropout yang digunakan untuk mencegah terjadinya overfitting
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            #Variabel model ditambahkan fungsi Dense dengan parameter variabel num_classes menggunakan activation softmax
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #Variabel model di compile dengan parameter loss, matrik, dan optimasi
            log_dir = '.\\logs\\conv2d_%d-dense_%d-dropout_%.2f' % (conv2d_count, dense_size, dropout)
            #Melakukan log pada dir 
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
            #Mengisi variabel tensorboard dengan isian dari library keras dan nilai dari log dir
            
            start = time.time()
            #Membuat variabel start dengan isi fungsi time dari library time
            model.fit(train_input, train_output, batch_size=32, epochs=10,
                      verbose=0, validation_split=0.2, callbacks=[tensorboard])
            #Melakukan fitting pada model dengan parameter test_input, test_output, batch_size,epochs,verbose,validation_split dan callbacks
            score = model.evaluate(test_input, test_output, verbose=2)
            #Membuat variabel score dengan nilai evaluasi dari model menggunakan data tes input dan tes output dengan verbose adalah 2
            end = time.time()
            #Membuat variabel end dengan isi fungsi time dari library time
            elapsed = end - start
            #Membuat variabel elapse yang diisi dengan nilai hasil waktu end dikurangi start
            print("Conv2D count: %d, Dense size: %d, Dropout: %.2f - Loss: %.2f, Accuracy: %.2f, Time: %d sec" % (conv2d_count, dense_size, dropout, score[0], score[1], elapsed))
            results.append((conv2d_count, dense_size, dropout, score[0], score[1], elapsed))
            #Menampilkan hasil perhitungan.

# In[14]:rebuild/retrain a model with the best parameters (from the search) and use all data
model = tf.keras.Sequential()
#Membuat variabel model dengan isi fungsi Sequential
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
#Variabel model ditambahkan fungsi Conv2d dengan paramater 32 filter dengan karnel berukuran 3x3 
#dengan algoritam activation relu mengunakan data train_input mulai dari baris nol.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#Variabel madel ditambahkan fungsi MaxPooling2D dengan ketentuan ukuran 2x2
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#Variabel model ditambahkan fungsi Conv2D dengan 32 filter dengan konvolusi berukuran 3x3, menggunakan algoritam activation relu
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#Variabel madel ditambahkan fungsi MaxPooling2D dengan ketentuan ukuran 2x2
model.add(tf.keras.layers.Flatten())
#Variabel model di tambahkan fungsi Flatten
model.add(tf.keras.layers.Dense(128, activation='tanh'))
#Variabel model ditambahakan fungsi dense dengan 128 neuron, dan menggunakan algoritma tanh untuk activation
model.add(tf.keras.layers.Dropout(0.5))
#Variabel model di tambahkan fungsi Dropout sebesar 50% untuk mencegah terjadinya overfitting
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
#Variabel model ditambahkan fungsi Dense dengan parameter variabel num_classes menggunakan activation softmax
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Variabel model di compile dengan parameter loss, matrik, dan optimasi
print(model.summary())
#Menampilkan ringkasan model yang telah dibuat.

# In[15]:join train and test data so we train the network on all data we have available to us
model.fit(np.concatenate((train_input, test_input)),
#Melakukan fitting pada model melakukan join numpy menggunakan data train_input test_input
          np.concatenate((train_output, test_output)),
          #kemudian join numpy menggunakan data train_output test_output
          batch_size=32, epochs=10, verbose=2)
          #menggunaakn batch ukuran 32, epochs 10 dan verbose 2

# In[16]:save the trained model
model.save("mathsymbols.model")
#Menyimpan model dengan nama mathsymbols.model

# In[17]:save label encoder (to reverse one-hot encoding)
np.save('classes.npy', label_encoder.classes_)
#Menyimpan label enkoder (untuk membalikkan one-hot encoder) dengan nama classes.npy

# In[18]:load the pre-trained model and predict the math symbol for an arbitrary image;
# the code below could be placed in a separate file

import keras.models
#Mengimport library keras.models
model2 = keras.models.load_model("mathsymbols.model")
#Membuat variabel model2 dengan isi hasil load model dari mathsymbols.model
print(model2.summary())
#Menampilkan ringkasan dari model

# In[19]:restore the class name to integer encoder
label_encoder2 = LabelEncoder()
#Membuat variabel label_encoder ke 2 dengan isi fungsi Label_Encoder
label_encoder2.classes_ = np.load('classes.npy')
#Menggunakan method classess dengan data classess.npy yang di eksport.

def predict(img_path):
    #Membuat fungsi predict dengan path img
    newimg = keras.preprocessing.image.img_to_array(pil_image.open(img_path))
    #Membuat variable newping dengan isi mengubah bentuk image menjadi array dan membuka data berdasarkan img path
    newimg /= 255.0
    #Membagi data yang terdapat pada newimg dengan 255.0

    # do the prediction
    prediction = model2.predict(newimg.reshape(1, 32, 32, 3))
    #Membuat variabel prediction dengan isian variabel model2 menggunakan fungsi predict dengan syarat variabel newimg dengan data reshape

    # figure out which output neuron had the highest score, and reverse the one-hot encoding
    inverted = label_encoder2.inverse_transform([np.argmax(prediction)]) # argmax finds highest-scoring output
    #Membuat variabel inverted  denagan label encoder2 dan  menggunakan argmax untuk mencari skor luaran tertinggi
    print("Prediction: %s, confidence: %.2f" % (inverted[0], np.max(prediction)))
    #Menampilkan prediksi gambar dan confidence dari gambar.
    
# In[20]: grab an image (we'll just use a random training image for demonstration purposes)
predict("D:/New folder/KB3C/src/1174084/7/HASYv2/hasy-data/v2-00010.png")
#Melakukan prediksi dari pelatihan dari gambar v2-00010.png
predict("D:/New folder/KB3C/src/1174084/7/HASYv2/hasy-data/v2-00500.png")
#Melakukan prediksi dari pelatihan dari gambar v2-00500.png
predict("D:/New folder/KB3C/src/1174084/7/HASYv2/hasy-data/v2-00700.png")
#Melakukan prediksi dari pelatihan dari gambar v2-00700.png
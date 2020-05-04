# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:17:40 2020

@author: Rin
"""

# In[1]:import lib
import csv #Import library csv
from PIL import Image as pil_image #Import library Image yaitu fungsi PIL (Python Imaging Library) yang berguna untuk mengolah data berupa gambar
import keras.preprocessing.image #Import library keras yang menggunakan method preprocessing yang digunakan untuk membuat neural network

# In[2]:load all images (as numpy arrays) and save their classes
imgs = [] #Membuat variabel imgs
classes = [] #Membuat variabel classes dengan variabel kosong
with open('N:/HASYv2/hasy-data-labels.csv') as csvfile: #Membuka file hasy-data-labels.csv
    csvreader = csv.reader(csvfile) #Membuat variabel csvreader yang berisi method csv.reader untuk membaca csvfile 
    i = 0 # membuat variabel i dengan isi 0
    for row in csvreader: # Membuat looping pada variabel csvreader
        if i > 0: #Ketentuannya jika i lebih kecil daripada 0
            img = keras.preprocessing.image.img_to_array(pil_image.open("N:/HASYv2/" + row[0])) #Dibuat variabel img dengan isi keras untuk aktivasi neural network fungsi yang membaca data yang berada dalam folder HASYv2 dengan input nilai -1.0 dan 1.0
            # neuron activation functions behave best when input values are between 0.0 and 1.0 (or -1.0 and 1.0),
            # so we rescale each pixel value to be in the range 0.0 to 1.0 instead of 0-255
            img /= 255.0 #Membagi data yang ada pada fungsi img sebanyak 255.0
            imgs.append((row[0], row[2], img)) #Menambah nilai baru pada imgs pada row ke 1 2 dan dilanjutkan dengan variabel img
            classes.append(row[2]) #Menambahkan nilai pada row ke 2 pada variabel classes
        i += 1 #Menambah nilai satu pada variabel i
        
# In[3]:shuffle the data, split into 80% train, 20% test
import random #Mengimport library random 
random.shuffle(imgs) #Melakukan randomize pada fungsi imgs
split_idx = int(0.8*len(imgs)) #Membuat variabel split_idx dengan nilai integer 80 persen dikali dari pengembalian jumlah dari variabel imgs
train = imgs[:split_idx] #Membuat variabel train dengan isi sebelum split idx
test = imgs[split_idx:] #Membuat variabel test dengan isi setelah split idx

# In[4]: 
import numpy as np #Mengimport library numpy dengan inisial np
train_input = np.asarray(list(map(lambda row: row[2], train))) #Membuat variabel train input dengan np method asarray yang mana membuat array dengan isi row 2 dari data train
test_input = np.asarray(list(map(lambda row: row[2], test))) #Membuat test input input dengan np method asarray yang mana membuat array dengan isi row 2 dari data test
train_output = np.asarray(list(map(lambda row: row[1], train))) #Membuat variabel train_output dengan np method asarray yang mana membuat array dengan isi row 1 dari data train
test_output = np.asarray(list(map(lambda row: row[1], test))) #Membuat variabel test_output dengan np method asarray yang mana membuat array dengan isi row 1 dari data test

# In[5]: import encoder and one hot
from sklearn.preprocessing import LabelEncoder #Mengimport library LabelEncode dari sklearn
from sklearn.preprocessing import OneHotEncoder #Mengimport library OneHotEncoder dari sklearn

# In[6]:convert class names into one-hot encoding
label_encoder = LabelEncoder() #Membuat variabel label_encoder dengan isi LabelEncoder
integer_encoded = label_encoder.fit_transform(classes) #Membuat variabel integer_encoded yang berfungsi untuk mengkonvert variabel classes kedalam bentuk integer

# In[7]:then convert integers into one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)#Membuat variabel onehot_encoder dengan isi OneHotEncoder
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) #Mengisi variabel integer_encoded dengan isi integer_encoded yang telah di convert pada fungsi sebelumnya
onehot_encoder.fit(integer_encoded) #Mengkonvert variabel integer_encoded kedalam onehot_encoder

# In[8]:convert train and test output to one-hot
train_output_int = label_encoder.transform(train_output) #Mengkonvert data train output  mengguanakn variabel label_encoder kedalam variabel train_output_int
train_output = onehot_encoder.transform(train_output_int.reshape(len(train_output_int), 1)) #Mengkonvert variabel train_output_int kedalam fungsi onehot_encoder 
test_output_int = label_encoder.transform(test_output) #Mengkonvert data test_output mengguanakn variabel label_encoder kedalam variabel test_output_int
test_output = onehot_encoder.transform(test_output_int.reshape(len(test_output_int), 1)) #Mengkonvert variabel test_output_int kedalam fungsi onehot_encoder 
num_classes = len(label_encoder.classes_) #Membuat variabel num_classes dengan isi variabel label_encoder dan classess
print("Number of classes: %d" % num_classes) #Mencetak hasil dari nomer Class beruapa persen 

# In[9]: import sequential
from keras.models import Sequential #Mengimport library Sequential dari Keras
from keras.layers import Dense, Dropout, Flatten #Mengimport library Dense, Dropout, Flatten dari Keras
from keras.layers import Conv2D, MaxPooling2D #Mengimport library Conv2D, MaxPooling2D dari Keras
#import tensorflow #Import tensorflow

# In[10]: desain jaringan
model = Sequential() #Membuat variabel model dengan isian library Sequential
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=np.shape(train_input[0]))) #Variabel model di tambahkan library Conv2D tigapuluh dua bit dengan ukuran kernel 3 x 3 dan fungsi penghitungan relu dang menggunakan data train_input
model.add(MaxPooling2D(pool_size=(2, 2))) #Variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixel 
model.add(Conv2D(32, (3, 3), activation='relu')) #Variabel model di tambahkan dengan library Conv2D 32bit dengan kernel 3 x 3
model.add(MaxPooling2D(pool_size=(2, 2))) #Variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixcel 
model.add(Flatten()) #Variabel model di tambahkan library Flatten
model.add(Dense(1024, activation='tanh')) #Variabel model di tambahkan library Dense dengan fungsi tanh
model.add(Dropout(0.5)) #Variabel model di tambahkan library dropout untuk memangkas data tree sebesar 50 persen
model.add(Dense(num_classes, activation='softmax')) #Variabel model di tambahkan library Dense dengan data dari num_classes dan fungsi softmax
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) #Mengkompile data model untuk mendapatkan data loss akurasi dan optimasi
print(model.summary()) #Mencetak variabel model kemudian memunculkan kesimpulan berupa data total parameter, trainable paremeter dan bukan trainable parameter

# In[11]: import sequential
import keras.callbacks #Mengimport library keras dengan fungsi callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir='N:/KB/hasyv2/logs/mnist-style') #Membuat variabel tensorboard dengan isi lib keras 

# In[12]: 5menit kali 10 epoch = 50 menit
model.fit(train_input, train_output, #Fungsi model ditambahkan fungsi fit untuk mengetahui perhitungan dari train_input train_output
          batch_size=32, #Dengan batch size 32 bit 
          epochs=10,
          verbose=2,
          validation_split=0.2,
          callbacks=[tensorboard])

score = model.evaluate(test_input, test_output, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# In[13]:try various model configurations and parameters to find the best
import time #Mengimport library time 
results = [] #Membuat variabel result dengan array kosong
for conv2d_count in [1, 2]: #Melakukan looping dengan ketentuan konvolusi 2 dimensi 1 2 
    for dense_size in [128, 256, 512, 1024, 2048]: #Menentukan ukuran besaran fixcel dari data atau konvert 1 fixcel mnjadi data yang berada pada codigan dibawah.
        for dropout in [0.0, 0.25, 0.50, 0.75]: #Membuat looping untuk memangkas masing-masing data dengan ketentuan 0 persen 25 persen 50 persen dan 75 persen.
            model = Sequential() #Membuat variabel model Sequential
            for i in range(conv2d_count): #Membuat looping untuk variabel i dengan jarak dari hasil konvolusi.
                if i == 0: #Syarat jika i samadengan bobotnya 0
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0]))) #Menambahkan method add pada variabel model dengan konvolusi 2 dimensi 32 bit didalamnya dan membuat kernel dengan ukuran 3 x 3 dan rumus aktifasi relu dan data shape yang di hitung dari data train.
                else: #Jika tidak
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))  #Menambahkan method add pada variabel model dengan konvolusi 2 dimensi 32 bit dengan ukuran kernel 3 x3 dan fungsi aktivasi relu
                model.add(MaxPooling2D(pool_size=(2, 2))) #Menambahkan method add pada variabel model dengan isian method  Max pooling berdimensi 2 dengan ukuran fixcel 2 x 2.
            model.add(Flatten()) #Merubah feature gambar menjadi 1 dimensi vektor
            model.add(Dense(dense_size, activation='tanh')) #Menambahkan method dense untuk pemadatan data dengan ukuran dense di tentukan dengan rumus fungsi tanh.
            if dropout > 0.0: #Membuat ketentuan jika pemangkasan lebih besar dari 0 persen
                model.add(Dropout(dropout)) #Menambahkan method dropout pada model dengan nilai dari dropout
            model.add(Dense(num_classes, activation='softmax')) #Menambahkan method dense dengan fungsi num classs dan rumus softmax
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #Mengcompile variabel model dengan hasi loss optimasi dan akurasi matrix
            log_dir = 'N:/KB/hasyv2/logs/conv2d_%d-dense_%d-dropout_%.2f' % (conv2d_count, dense_size, dropout) #Melakukan log
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir) # membuat variabel tensorboard dengan isian dari library keras dan nilai dari log_dir

            start = time.time() #Membuat variabel start dengan isian dari library time menggunakan method time
            model.fit(train_input, train_output, batch_size=32, epochs=10,
                      verbose=0, validation_split=0.2, callbacks=[tensorboard]) #Menambahkan method fit pada model dengan data dari train input train output nilai batch nilai epoch verbose nilai 20 persen validation split dan callback dengan nilai tensorboard.
            score = model.evaluate(test_input, test_output, verbose=2) #Membuat variabel score dengan nilai evaluasi dari model menggunakan data tes input dan tes output
            end = time.time() #Membuat variabel end 
            elapsed = end - start  #Membuat variabel elapsed
            print("Conv2D count: %d, Dense size: %d, Dropout: %.2f - Loss: %.2f, Accuracy: %.2f, Time: %d sec" % (conv2d_count, dense_size, dropout, score[0], score[1], elapsed)) #Mencetak hasil perhitungan
            results.append((conv2d_count, dense_size, dropout, score[0], score[1], elapsed))

# In[14]:rebuild/retrain a model with the best parameters (from the search) and use all data
model = Sequential() #Membuat variabel model dengan isian library Sequential
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))#Variabel model di tambahkan library Conv2D tigapuluh dua bit dengan ukuran kernel 3 x 3 dan fungsi penghitungan relu dang menggunakan data train_input
model.add(MaxPooling2D(pool_size=(2, 2))) #Variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixel
model.add(Conv2D(32, (3, 3), activation='relu')) #Variabel model di tambahkan dengan library Conv2D 32bit dengan kernel 3 x 3
model.add(MaxPooling2D(pool_size=(2, 2))) #Variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixcel
model.add(Flatten()) #Variabel model di tambahkan library Flatten
model.add(Dense(128, activation='tanh')) #Variabel model di tambahkan library Dense dengan fungsi tanh
model.add(Dropout(0.5)) #Variabel model di tambahkan library dropout untuk memangkas data tree sebesar 50 persen
model.add(Dense(num_classes, activation='softmax')) #Variabel model di tambahkan library Dense dengan data dari num_classes dan fungsi softmax
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #Mengkompile data model untuk mendapatkan data loss akurasi dan optimasi
print(model.summary()) #Mencetak variabel model kemudian memunculkan kesimpulan berupa data total parameter, trainable paremeter dan bukan trainable parameter

# In[15]:join train and test data so we train the network on all data we have available to us
model.fit(np.concatenate((train_input, test_input)), #Melakukan training yang isi datanya dari join numpy menggunakan data train_input test_input
          np.concatenate((train_output, test_output)), #Kelanjutan data yang di gunakan pada join train_output test_output
          batch_size=32, epochs=10, verbose=2) #Menggunakan ukuran 32 bit dan epoch 10 

# In[16]:save the trained model
model.save("mathsymbols.model") #Menyimpan model atau mengeksport model yang telah di jalan tadi

# In[17]:save label encoder (to reverse one-hot encoding)
np.save('classes.npy', label_encoder.classes_) #Menyimpan label encoder dengan nama classes.npy 

# In[18]:load the pre-trained model and predict the math symbol for an arbitrary image;
# the code below could be placed in a separate file
import keras.models #Mengimpport library keras model
model2 = keras.models.load_model("mathsymbols.model") #Membuat variabel model2 untuk meload model yang telah di simpan tadi
print(model2.summary()) #Mencetak hasil model2

# In[19]:restore the class name to integer encoder
label_encoder2 = LabelEncoder() # membuat variabel label encoder ke 2 dengan isian fungsi label encoder.
label_encoder2.classes_ = np.load('classes.npy') #Menambahkan method classess dengan data classess yang di eksport tadi
def predict(img_path): #Membuat fumgsi predict dengan path img
    newimg = keras.preprocessing.image.img_to_array(pil_image.open(img_path)) #Membuat variabel newimg dengam membuay immage menjadi array dan membuka data berdasarkan img path
    newimg /= 255.0 #Membagi data yang terdapat pada variabel newimg sebanyak 255

    # do the prediction
    prediction = model2.predict(newimg.reshape(1, 32, 32, 3)) #Membuat variabel predivtion dengan isian variabel model2 menggunakan fungsi predic dengan syarat variabel newimg dengan data reshape

    # figure out which output neuron had the highest score, and reverse the one-hot encoding
    inverted = label_encoder2.inverse_transform([np.argmax(prediction)]) #Membuat variabel inverted  denagan label encoder2 dan  menggunakan argmax untuk mencari skor keluaran tertinggi
    print("Prediction: %s, confidence: %.2f" % (inverted[0], np.max(prediction))) #Mencetak prediksi gambar dan confidence dari gambar.

# In[20]: grab an image (we'll just use a random training image for demonstration purposes)
predict("HASYv2/hasy-data/v2-00010.png") #Mencari prediksi menggunakan fungsi prediksi yang di buat tadi dari data di HASYv2/hasy-data/v2-00010.png
predict("HASYv2/hasy-data/v2-00500.png") #Mencari prediksi menggunakan fungsi prediksi yang di buat tadi dari data di HASYv2/hasy-data/v2-00500.png
predict("HASYv2/hasy-data/v2-00700.png") #Mencari prediksi menggunakan fungsi prediksi yang di buat tadi dari data di HASYv2/hasy-data/v2-00700.png
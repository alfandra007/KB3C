import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

# Membuat fungsi yang akan menampilkan grafik
def display_mfcc(song):
    # y akan meload variabel song
    y, _ = librosa.load(song)
	# Konversi audio menjadi bentuk vektor
    mfcc = librosa.feature.mfcc(y)
	# membuat Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()

# Gunakan fungsi dengan parameter file musik
display_mfcc('fallOutBoy/Fall Out Boy - Uma Thurman.mp3')
display_mfcc('adele/Adele - Make You Feel My Love.mp3')


def extract_features_song(f):
	# y akan meload variabel f
    y, _ = librosa.load(f)
	# Konversi audio menjadi bentuk vektor
    mfcc = librosa.feature.mfcc(y)
	# mff akan dibagi oleh numpy amax
    mfcc /= np.amax(np.absolute(mfcc))
	# Tampilkan dalam bentuk array 25000
	# Melakukan data training dengan mengambil dengan durasi yang sama
    return np.ndarray.flatten(mfcc)[:25000]

extract_features_song('fallOutBoy/Fall Out Boy - Uma Thurman.mp3')

extract_features_song('adele/Adele - Make You Feel My Love.mp3')

def generate_features_and_labels():
	# Buat array kosong, yang nantinya akan dipush data
    all_features = []
    all_labels = []
	
	# Sesuai dengan nama penyanyi
    penyanyi = ['adele', 'brunoMars','edSheeran', 'fallOutBoy', 
	'jasonMraz','maroonFive', 'meghanTrainor', 'michaelJackson', 
	'oneRepublic', 'theChainsmokers']
	# Looping
    for p in penyanyi:
		# Mengambil file dari folder penyanyi dan semua ekstensi mp3
        sound_files = glob.glob(p+'/*.mp3')
		# Jumlah Lagu dan Penyanyinya siapa
        print('Processing %d songs by %s ...' % (len(sound_files), p))
        
        for f in sound_files:
			# Memanggil function dan mengirimkan 
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(p)

    # convert labels to one-hot encoding
	# Convert agar bisa dimengerti meachine learning
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    # Mengubah ke bentuk one hot encoding
	onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    # Mengembalikan variabel all_features & onehot_labels kedalam satu matrix
	return np.stack(all_features), onehot_labels

# Proses ini akan sangat lama, karena akan dilakukan features dan perubahan label pada setiap folder
# Tampung Kedalam variabel all_features kedalam features, onehot_labels kedalam labels
features, labels = generate_features_and_labels()
# Dimensi dari variabel features & labels
print('Dimensi Features :', np.shape(features))
print('Dimensi labels :', np.shape(labels))

# Split jadi 80% data training
training_split = 0.8
# Melakukan penumpukan features dan labels
alldata = np.column_stack((features, labels))
# Melakukan pengocokan 
np.random.shuffle(alldata)
# Kalikan panjang dari angka alldata & training_split
splitidx = int(len(alldata) * training_split)
# Pisahkan data training & data test
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

print('Data Train :', np.shape(train))

print('Data test :'np.shape(test))

# kecualikan 10 baris terakhir
train_input = train[:,:-10]

test_input = test[:,:-10]

# mengambil 10 baris saja
train_labels = train[:,-10:]
test_labels = test[:,-10:]

print('Dimensi Train Input :',np.shape(train_input))
print('Dimensi Labels Train :',np.shape(train_labels))

print('Dimensi Train Input :',np.shape(test_input))
print('Dimensi Labels Train :',np.shape(test_labels))

# Menentukan Model pembelajaran untuk meachine learning
model = tf.keras.Sequential()
# Layer pertama dense dari 100 neuron untuk inputan
model.add(tf.keras.layers.Dense(100, input_dim=np.shape(train_input)[1]))
# Activation menggunakan fungsi relu, pilih inputan dengan nilai maksimum
model.add(tf.keras.layers.Activation('relu'))
# Dense Kategorikan 10 neuron untuk jenis penyanyi
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))

# Pake algoritma adam sebagai optimizer
# memperbaui bobot jaringan yang beurlang berdasarkan data training
model.compile(optimizer='adam',
# Loss categorical_crossentropy untuk optimasi skor
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# Lakukan training
model.fit(train_input, train_labels, 
# epochs, rambatan baliknya sebanyak 10
epochs=10, 
# dalam sekali epochs dilakukan 32 sampel sebelum model itu diperbarui
batch_size=32,
# cek cross score
validation_split=0.2)

# Akan menghasilkan Loss(hasil prediksi yang salah) dan accurasi
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# save the trained model
model.save("penyanyi.hdf5")

import tensorflow as tf 
# Load model
model = tf.keras.models.load_model("penyanyi.hdf5")
print(model.summary())

# Membuat prediksi dengan model sebelumnya
def predict(song_path):
    song = np.stack([extract_features_song(song_path)])
    # do the prediction
    prediction = model2.predict(song, batch_size=32)
	
    print("Prediction: %s, confidence: %.2f" % (np.argmax(prediction), np.max(prediction)))

predict('theChainsmokers/The Chainsmokers - All We Know.mp3')

predict('edSheeran/Ed Sheeran - Castle On The Hill.mp3')

# Buat Confusion matrix
from sklearn.metrics import confusion_matrixs
pred_labels = model.predict(test_input)
cm = confusion_matrix(test_labels.argmax(axis=1), pred_labels.argmax(axis=1))
cm

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
	
    plt.figure(figsize=(6,6), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

penyanyi = ['adele', 'brunoMars','edSheeran', 'fallOutBoy', 
	'jasonMraz','maroonFive', 'meghanTrainor', 'michaelJackson', 
	'oneRepublic', 'theChainsmokers']
plot_confusion_matrix(cm, classes=penyanyi, normalize=True)
plt.show()
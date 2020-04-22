# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:20:09 2020

@author: PandA23
"""

# In[]

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

# In[]
def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()
# In[]
display_mfcc('singers/Agnes_Monica/Agnes Monica - Bukan Milikmu Lagi.mp3')
# In[]
display_mfcc('singers/Bunga_Citra_Lestari/Bunga Citra Lestari - Aku_Bisa_Apa.mp3')
# In[]
def extract_features_song(f):
    y, _ = librosa.load(f)
    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))
    
    return np.ndarray.flatten(mfcc)[:25000]

# In[]
extract_features_song('singers/Agnes_Monica/Agnes Monica - Bukan Milikmu Lagi.mp3')
# In[]
extract_features_song('singers/Bunga_Citra_Lestari/Bunga Citra Lestari - Aku_Bisa_Apa.mp3')
# In[]
def generate_features_and_labels():
    all_features = []
    all_labels = []

    penyanyis = ['Tulus', 'Opick', 'Bunga_Citra_Lestari', 'Marcell', 'Chrisye','Glenn_Fredly', 'Inul_Daratista', 'Judika', 'Nike_Ardila', 'Agnes_Monica']
    for penyanyi in penyanyis:
        sound_files = glob.glob('singers/'+penyanyi+'/*.mp3')
        print('Memproses %d lagu dari %s ...' % (len(sound_files), penyanyi))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(penyanyi)

    # convert labels to one-hot encoding cth blues : 1000000000 classic 0100000000
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)#ke integer
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))#ke one hot
    return np.stack(all_features), onehot_labels

# In[]
features, labels = generate_features_and_labels()
# In[]
print(np.shape(features))
print(np.shape(labels))
# In[]
training_split = 0.8
# In[]
alldata = np.column_stack((features, labels))
# In[]
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
# In[]
print(np.shape(train))
print(np.shape(test))
# In[]
train_input = train[:,:-10]
train_labels = train[:,-10:]
# In[]
test_input = test[:,:-10]
test_labels = test[:,-10:]
# In[]
print(np.shape(train_input))
print(np.shape(train_labels))

# In[]:
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, input_dim=np.shape(train_input)[1]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
# In[]:
model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))
# In[]:
# save the trained model
model.save("penyanyi.hdf5")
# In[]:
import tensorflow as tf 
model2 = tf.keras.models.load_model("penyanyi.hdf5")
print(model2.summary())
# In[]:
def predict(song_path):
    song = np.stack([extract_features_song(song_path)])
    # do the prediction
    prediction = model2.predict(song, batch_size=32)

    print("Prediction: %s, confidence: %.2f" % (np.argmax(prediction), np.max(prediction)))
# In[]:
predict('singers/Agnes_Monica/Agnes Monica - Bukan Milikmu Lagi.mp3')
# In[]:
predict('singers/Bunga_Citra_Lestari/Bunga Citra Lestari - Aku_Bisa_Apa.mp3')
# In[]:
from sklearn.metrics import confusion_matrix
pred_labels = model2.predict(test_input)
cm = confusion_matrix(test_labels.argmax(axis=1), pred_labels.argmax(axis=1))
cm
# In[]:
import matplotlib.pyplot as plt
import itertools
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
    plt.figure(figsize=(6,6), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# In[]:
import numpy as np
singers = ['Tulus', 'Opick', 'Bunga_Citra_Lestari', 'Marcell', 'Chrisye','Glenn_Fredly', 'Inul_Daratista', 'Judika', 'Nike_Ardila', 'Agnes_Monica']
plot_confusion_matrix(cm, classes=singers, normalize=True)
plt.show()
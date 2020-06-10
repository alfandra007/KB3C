# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:04:52 2020

@author: FannyShafira
"""
# In[1]: Soal Nomor 1

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()
# In[2]: Soal Nomor 2
display_mfcc('test1.wav')
display_mfcc('test2.wav')
display_mfcc('dataset/genres/disco/disco.00069.au')
display_mfcc('dataset/genres/blues/blues.00069.au')
display_mfcc('dataset/genres/classical/classical.00069.au')
display_mfcc('dataset/genres/country/country.00069.au')
display_mfcc('dataset/genres/hiphop/hiphop.00069.au')
display_mfcc('dataset/genres/jazz/jazz.00069.au')
display_mfcc('dataset/genres/pop/pop.00069.au')
display_mfcc('dataset/genres/reggae/reggae.00069.au')
display_mfcc('dataset/genres/rock/rock.00069.au')
# In[3]: Soal Nomor 3

def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

# In[4]: Soal Nomor 4

def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob('dataset/genres/'+genre+'/*.au')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding cth blues : 1000000000 classic 0100000000
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)#ke integer
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))#ke one hot
    return np.stack(all_features), onehot_labels

# In[5]: Soal Nomor 5

features, labels = generate_features_and_labels()
print(np.shape(features))
print(np.shape(labels))
# In[6]: Soal Nomor 6
training_split = 0.8
alldata = np.column_stack((features, labels))
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
print(np.shape(train))
print(np.shape(test))
train_input = train[:,:-10]
train_labels = train[:,-10:]
test_input = test[:,:-10]
test_labels = test[:,-10:]
print(np.shape(train_input))
print(np.shape(train_labels))

# In[7]: Soal Nomor 7
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
# In[8]: Soal Nomor 8
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
# In[9]: Soal Nomor 9
model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)
# In[10]: Soal Nomor 10
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# In[11]: Soal Nomor 11
model.predict(test_input[:1])
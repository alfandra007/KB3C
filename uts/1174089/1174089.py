 # In[]:
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

# In[]:
def display_mfcc(song): #membuat fungsi display_mfcc
    y, _ = librosa.load(song) # variabel Y yang berisi method librosa load 
    mfcc = librosa.feature.mfcc(y) # method librosa featurea mfcc

    plt.figure(figsize=(10, 4)) #membuat ﬂot ﬁgure dengan ukuran 10 banding 4
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel') #data librosa display dengan variabel x nya yaitu waktu dan y yaitu mel atau Hz
    plt.colorbar() #plot bar warna
    plt.title(song) #judul plot 
    plt.tight_layout()
    plt.show()

# In[]:
display_mfcc('D:/New folder/KB3C/src/1174089/uts/singers/The Dramma/Sanji Indrajaya - Cinta Terjauh Ku (Official Music Video).mp3') #mendisplay tampilan gelombang suara 
# In[]:
display_mfcc('D:/New folder/KB3C/src/1174089/uts/singers/Last Child/Last Child - Bernafas Tanpamu (Official Lyric Video).mp3')
# In[]:
def extract_features_song(f):
    y, _ = librosa.load(f)
    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[:25000]
# In[]:
extract_features_song('D:/New folder/KB3C/src/1174089/uts/singers/The Dramma/Sanji Indrajaya - Cinta Terjauh Ku (Official Music Video).mp3')
# In[]:
extract_features_song('D:/New folder/KB3C/src/1174089/uts/singers/Last Child/Last Child - Bernafas Tanpamu (Official Lyric Video).mp3')
# In[]:
def generate_features_and_labels():
    all_features = []
    all_labels = []

    singers = ['The Dramma', 'Last Child', 'Brad Paisley', 'Breaking Benjamin', 'Christina Peri', 'Dream Theater', 'Jorja Smith', 'Katelyn Tarver', 'Scorpion', 'Thrisha yearwood']
    for singer in singers:
        sound_files = glob.glob('D:/New folder/KB3C/src/1174089/uts/singers/'+singer+'/*.mp3') #mengembalikan array file / direktori
        print('Processing %d songs by %s ...' % (len(sound_files), singer))
        
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(singer)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), onehot_labels
# In[]:
features, labels = generate_features_and_labels()
# In[]:
print(np.shape(features))
print(np.shape(labels))
# In[]:
training_split = 0.8
# In[]:
# last column has singer, turn it into unique ids
alldata = np.column_stack((features, labels))
# In[]:
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
# In[]:
print(np.shape(train))
print(np.shape(test))
# In[]:
train_input = train[:,:-10]
train_labels = train[:,-10:]

print(np.shape(train_input))
print(np.shape(train_labels))
# In[]:
test_input = test[:,:-10]
test_labels = test[:,-10:]

print(np.shape(test_input))
print(np.shape(test_labels))

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
model.save("advent.singers.hdf5")
# In[]:
import tensorflow as tf 
model2 = tf.keras.models.load_model("advent.singers.hdf5")
print(model2.summary())
# In[]:
def predict(song_path):
    song = np.stack([extract_features_song(song_path)])
    # do the prediction
    prediction = model2.predict(song, batch_size=32)

    print("Prediction: %s, confidence: %.2f" % (np.argmax(prediction), np.max(prediction)))
# In[]:
predict('D:/New folder/KB3C/src/1174089/uts/singers/Dream Theater/Dream Theater - Acoustic Dreams FULL Album.mp3')
# In[]:
predict('D:/New folder/KB3C/src/1174089/uts/singers/Brad Paisley/Brad-Paisley-All-In.mp3')
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
singers = ['The Dramma', 'Last Child', 'Brad Paisley', 'Breaking Benjamin', 'Christina Peri', 'Dream Theater', 'Jorja Smith', 'Katelyn Tarver', 'Scorpion', 'Thrisha yearwood']
plot_confusion_matrix(cm, classes=singers, normalize=True)
plt.show()
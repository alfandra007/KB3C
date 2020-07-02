# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:04:40 2020

@author: Aulyardha Anindita
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, Activation, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
from scipy.io import loadmat

# In[]: Nomor 1
from google.colab import drive
drive.mount('/content/drive')

import tarfile
tf = tarfile.open("/content/drive/My Drive/Chapter 9 AI/wiki_crop.tar")
tf.extractall(path="/content/drive/My Drive/Chapter 9 AI")

# In[]: Nomor 2
def load_data(wiki_dir, dataset='wiki'):
    # Load the wiki.mat file
    meta = loadmat(os.path.join(wiki_dir, "{}.mat".format(dataset)))

    # Load the list of all files
    full_path = meta[dataset][0, 0]["full_path"][0]

    # List of Matlab serial date numbers
    dob = meta[dataset][0, 0]["dob"][0]

    # List of years when photo was taken
    photo_taken = meta[dataset][0, 0]["photo_taken"][0]  # year

    # Calculate age for all dobs
    age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # Create a list of tuples containing a pair of an image path and age
    images = []
    age_list = []
    for index, image_path in enumerate(full_path):
        images.append(image_path[0])
        age_list.append(age[index])

    # Return a list of all images and respective age
    return images, age_list

# In[]: Nomor 3
def build_encoder():
    """
    Encoder Network
    """
    input_layer = Input(shape=(64, 64, 3))

    # 1st Convolutional Block
    enc = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(input_layer)
    # enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 2nd Convolutional Block
    enc = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 3rd Convolutional Block
    enc = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 4th Convolutional Block
    enc = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Flatten layer
    enc = Flatten()(enc)

    # 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Second Fully Connected Layer
    enc = Dense(100)(enc)

    # Create a model
    model = Model(inputs=[input_layer], outputs=[enc])
    return model  

# In[]: Nomor 4
def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    """
    latent_dims = 100
    num_classes = 6

    input_z_noise = Input(shape=(latent_dims,))
    input_label = Input(shape=(num_classes,))

    x = concatenate([input_z_noise, input_label])

    x = Dense(2048, input_dim=latent_dims + num_classes)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_z_noise, input_label], outputs=[x])
    return model

# In[] : Nomor 5
def build_discriminator():
    """
    Create a Discriminator Model with hyperparameters values defined as follows
    """
    input_shape = (64, 64, 3)
    label_shape = (6,)
    image_input = Input(shape=input_shape)
    label_input = Input(shape=label_shape)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    label_input1 = Lambda(expand_label_input)(label_input)
    x = concatenate([x, label_input1], axis=3)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, label_input], outputs=[x])
    return model

# In[] : Nomor 6
if __name__ == '__main__':
    # Define hyperparameters
    data_dir = "data"
    wiki_dir = os.path.join(data_dir, "wiki_crop1")
    epochs = 500
    batch_size = 2
    image_shape = (64, 64, 3)
    z_shape = 100
    TRAIN_GAN = True
    TRAIN_ENCODER = False
    TRAIN_GAN_WITH_FR = False
    fr_image_shape = (192, 192, 3)

    # Define optimizers
    dis_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    gen_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adversarial_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    """
    Build and compile networks
    """
    # Build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'], optimizer=dis_optimizer)

    # Build and compile the generator network
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    # Build and compile the adversarial model
    discriminator.trainable = False
    input_z_noise = Input(shape=(100,))
    input_label = Input(shape=(6,))
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])
    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    Load the dataset
    """
    images, age_list = load_data(wiki_dir=wiki_dir, dataset="wiki")
    age_cat = age_to_category(age_list)
    final_age_cat = np.reshape(np.array(age_cat), [len(age_cat), 1])
    classes = len(set(age_cat))
    y = to_categorical(final_age_cat, num_classes=len(set(age_cat)))

    loaded_images = load_images(wiki_dir, images, (image_shape[0], image_shape[1]))

    # Implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    """
    Train the generator and the discriminator network
    """
    if TRAIN_GAN:
        for epoch in range(epochs):
            print("Epoch:{}".format(epoch))

            gen_losses = []
            dis_losses = []

            number_of_batches = int(len(loaded_images) / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:{}".format(index + 1))

                images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[index * batch_size:(index + 1) * batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                """
                Train the discriminator network
                """

                # Generate fake images
                initial_recon_images = generator.predict_on_batch([z_noise, y_batch])

                d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = discriminator.train_on_batch([initial_recon_images, y_batch], fake_labels)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                print("d_loss:{}".format(d_loss))

                """
                Train the generator network
                """

                z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
                random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                random_labels = to_categorical(random_labels, 6)

                g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], [1] * batch_size)

                print("g_loss:{}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

            # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

            """
            Generate images after every 10th epoch
            """
            if epoch % 10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    save_rgb_img(img, path="results/img_{}_{}.png".format(epoch, i))

        # Save networks
        try:
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")
        except Exception as e:
            print("Error:", e)

# In[] : Nomor 7
if TRAIN_ENCODER:
        # Build and compile encoder
        encoder = build_encoder()
        encoder.compile(loss=euclidean_distance_loss, optimizer='adam')

        # Load the generator network's weights
        try:
            generator.load_weights("generator.h5")
        except Exception as e:
            print("Error:", e)

        z_i = np.random.normal(0, 1, size=(5000, z_shape))

        y = np.random.randint(low=0, high=6, size=(5000,), dtype=np.int64)
        num_classes = len(set(y))
        y = np.reshape(np.array(y), [len(y), 1])
        y = to_categorical(y, num_classes=num_classes)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            encoder_losses = []

            number_of_batches = int(z_i.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                z_batch = z_i[index * batch_size:(index + 1) * batch_size]
                y_batch = y[index * batch_size:(index + 1) * batch_size]

                generated_images = generator.predict_on_batch([z_batch, y_batch])

                # Train the encoder model
                encoder_loss = encoder.train_on_batch(generated_images, z_batch)
                print("Encoder loss:", encoder_loss)

                encoder_losses.append(encoder_loss)

            # Write the encoder loss to Tensorboard
            write_log(tensorboard, "encoder_loss", np.mean(encoder_losses), epoch)

        # Save the encoder model
        encoder.save_weights("encoder.h5")

#!/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = pathlib.Path('../data/spect')

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

DATASET_SIZE = 35690
TRAIN_SIZE = int(0.8 * DATASET_SIZE) 
VAL_SIZE = int(0.2 * DATASET_SIZE)

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 998

BATCH_SIZE = 128
EPOCHS = 10

def decode_img(img):
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def extract_label(file_path):
    path = tf.strings.split(file_path, os.path.sep)
    return path[-2] == CLASS_NAMES

def process_path(file_path):
    label = extract_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

ds_list = tf.data.Dataset.list_files(str(data_dir/'/*/*'))
ds_labelled = ds_list.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_labelled = ds_labelled.shuffle(DATASET_SIZE)
ds_train = ds_labelled.take(TRAIN_SIZE)
ds_val = ds_labelled.skip(TRAIN_SIZE)

ds_train = ds_train.batch(BATCH_SIZE)
ds_val = ds_val.batch(BATCH_SIZE)

#for image, label in ds_labelled.take(10):
#    print(image)
#    print(label)

model = Sequential([
    Conv2D(16, 2, padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
        ds_train.repeat(), 
        steps_per_epoch=(TRAIN_SIZE//BATCH_SIZE),
        epochs=EPOCHS, 
        validation_data=ds_val.repeat(), 
        validation_steps=(VAL_SIZE//BATCH_SIZE)
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

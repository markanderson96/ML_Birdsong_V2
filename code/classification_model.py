#!/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.executing_eagerly()

import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_SIZE = 35690
TRAIN_SIZE = int(0.8 * DATASET_SIZE) 
VAL_SIZE = int(0.2 * DATASET_SIZE)

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 1000

BATCH_SIZE = 64
epochs = 10

data_dir = '../data/spect'

image_generator = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=data_dir,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    subset='training',
    class_mode='binary'
)

val_gen = image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=data_dir,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    subset='validation',
    class_mode='binary'
)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
        train_gen, 
        steps_per_epoch=(TRAIN_SIZE//BATCH_SIZE),
        epochs=epochs, 
        validation_data=val_gen, 
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

#!/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.executing_eagerly()

from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_SIZE = 35690
TRAIN_SIZE = int(0.8 * DATASET_SIZE) 
VAL_SIZE = int(0.2 * DATASET_SIZE)

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 1000

BATCH_SIZE = 128
epochs = 12

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

logdir = "../logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = Sequential([
    Conv2D(16, 5, padding='valid', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(32, 5, padding='valid', activation='relu'),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(64, 5, padding='valid', activation='relu'),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(128, 5, padding='valid', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model.summary()

history = model.fit(
        train_gen, 
        steps_per_epoch=(TRAIN_SIZE//BATCH_SIZE),
        epochs=epochs, 
        validation_data=val_gen, 
        validation_steps=(VAL_SIZE//BATCH_SIZE),
        callbacks=[tensorboard_callback]
)

filename = datetime.now().strftime("%d%m%Y-%H%M%S") + '.h5'
model.save('../' + filename)

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

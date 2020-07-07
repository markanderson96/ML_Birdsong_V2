#!/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#tf.executing_eagerly()
#physical_devices = tf.config.list_physical_devices('GPU')
#try:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#except:
#    pass

from datetime import datetime
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import crnn

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def opts_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--type', type=str, default='linear')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-e', '--epochs', type=int, default=20)

    return parser

def main():
    parser = opts_parser()
    args = parser.parse_args()

    spect_type = args.type
    learning_rate = args.learning_rate
    epochs = args.epochs

    DATASET_SIZE = 35690
    TRAIN_SIZE = int(0.8 * DATASET_SIZE) 
    VAL_SIZE = int(0.2 * DATASET_SIZE)

    data_dir = '../data/spect'

    if spect_type == 'linear':
        IMAGE_HEIGHT = 257
        IMAGE_WIDTH = 1000
        BATCH_SIZE = 1 
    else:
        IMAGE_HEIGHT = 160
        IMAGE_WIDTH = 998
        BATCH_SIZE = 32 

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

    model = crnn.CRNN2D([IMAGE_HEIGHT, IMAGE_WIDTH, 3], 1)

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

if __name__ == "__main__":
    main()

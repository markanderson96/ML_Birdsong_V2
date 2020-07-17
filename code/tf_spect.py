#!/bin/python

import tensorflow as tf
import numpy as np
import os

from scipy import signal

tf.executing_eagerly()

data_root = '/home/mark/projects/PhD/datasets/DCASE2018'

for dirpath, dirs, files in os.walk(data_root):
    for filename in files:
        if(filename.endswith('.wav')):
            print('Making Spectrogram for {}'.format(dirpath+'/'+filename))
            audio_file = tf.io.read_file(dirpath + '/' + filename)
            waveform, sample_rate = tf.audio.decode_wav(audio_file)
        else: 
            continue

        waveform = np.array(waveform)
        waveform = waveform.reshape(waveform.shape[0])

        sample_rate = tf.cast(sample_rate, tf.float32)

        waveform = signal.decimate(waveform, 3)
        sample_rate = sample_rate//3

        frame_length = int(0.02*sample_rate)
        hop = int(frame_length/2)

        spectrogram = tf.signal.stft(waveform, frame_length, hop, window_fn=tf.signal.hamming_window, pad_end=True)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.divide(
                        tf.math.subtract(spectrogram, tf.math.reduce_min(spectrogram)),
                        tf.math.subtract(tf.math.reduce_max(spectrogram), tf.math.reduce_min(spectrogram))
                    )
        spectrogram = tf.multiply(spectrogram, 65535)
        spectrogram = tf.expand_dims(spectrogram, -1)
        
        spectrogram = tf.image.flip_left_right(spectrogram)
        spectrogram = tf.image.transpose(spectrogram)
        
        spect_out = tf.cast(spectrogram, tf.uint16)
        output_file = dirpath + '/' + filename[:-4] + '.png'
        tf.io.write_file(output_file, tf.image.encode_png(spect_out, compression=0))

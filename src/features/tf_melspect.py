from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import sys
import yaml
import queue
import logging
import os
from pathlib import Path

def log10(x):
    n = tf.math.log(x)
    d = tf.math.log(tf.constant(10, dtype=n.dtype))
    return(tf.divide(n,d))

def main():
    audio_path = input_loc + filename
    audio_file = tf.io.read_file(audio_path) # read audio file
    waveform, sample_rate = tf.audio.decode_wav(audio_file) # decode wav

    waveform = np.array(waveform) # reshape audio to remove channel dimension
    waveform = waveform.reshape(waveform.shape[0])

    sample_rate = tf.cast(sample_rate, tf.float32) # cast sr as float (needed for mel weighting)

    win_len = int(window_len * sample_rate)
    ov_len = int(overlap_len * sample_rate)

    # If using filtering
    if (spect_type == 'linear' and use_filter):
        Wl = min_freq / (sample_rate//2)
        Wh = max_freq / (sample_rate//2)
        b, a = signal.cheby2(4, 40, [Wl, Wh], 'bandpass')
        waveform = signal.lfilter(b, a, waveform)
    
    # If Preemphasis flag set
    if (preemphasis):
        waveform = signal.lfilter([1, -0.97], 1, waveform)

    stfts = tf.signal.stft(waveform, win_len, ov_len, window_fn=tf.signal.hann_window, pad_end=True) # take stft and absolute
    spectrograms = tf.abs(stfts)
    
    if (spect_type == 'mel'):
        num_fft_bins = stfts.shape[-1] # num fft bins
        mel_weights = tf.signal.linear_to_mel_weight_matrix(num_bands, num_fft_bins, sample_rate, min_freq, max_freq) # create filterbank
        spect = tf.tensordot(spectrograms, mel_weights, 1) # apply to stft
    else:
        spect = spectrograms

    # Log sacling and normalisation for image
    #spect = 10 * log10(tf.math.maximum(spect, 1E-06))
    spect = tf.math.divide(
               tf.math.subtract(spect, tf.math.reduce_min(spect)),
               tf.math.subtract(tf.math.reduce_max(spect), tf.math.reduce_min(spect))
        )
    spect = tf.multiply(spect, 255) # Change this to normalisation
    
    expanded = tf.expand_dims(spect, -1) # Needed to create image

    # Spectrogram is backwars and axes swapped, below code fixes this
    flipped = tf.image.flip_left_right(expanded)
    transposed = tf.image.transpose(flipped)

    # Cast to 8-bit unisgned
    out = tf.cast(transposed, tf.uint8)
    
    # Write out PNG of Spectrogram
    output_file = output_loc + filename[:-4] + '.png'
    tf.io.write_file(output_file, tf.image.encode_png(out, compression=0))

if __name__=="__main__":
    # filepath args
    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    
    # useful to have project dir
    project_dir = Path(__file__).resolve().parents[2]

    # logger formatting
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load config file
    with open(str(project_dir) + '/config/config.yaml') as file:
        config = yaml.safe_load(file)

    # dataset to process
    dataset = config['general']["dataset"]

    # input/output locations
    input_loc = input_filepath + os.sep + dataset + '/wav/train/'
    output_loc = output_filepath + os.sep + dataset + '/png/'
    if not os.path.isdir(output_loc):
        os.mkdir(output_loc)

    # create and fill queue
    file_queue = queue.Queue()
    for file in os.listdir(input_loc):
        file_queue.put(file)

    # load config vars
    min_freq = config['feature_creation']["min_freq"]
    max_freq = config['feature_creation']['max_freq']
    use_filter = config['feature_creation']['filter']
    downsample = config['feature_creation']['downsample']
    spect_type = config['feature_creation']["spect_type"]
    preemphasis = config['feature_creation']['preemphasis']
    window_len = config['feature_creation']['window_len']
    overlap_len = config['feature_creation']['overlap_len']
    num_bands = config['feature_creation']['num_bands']

    # start logger
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info('Spectrogram Type: {}'.format(spect_type))

    # Executes eagerly by default
    tf.executing_eagerly()
    logger.info("Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))

    while not file_queue.empty():
        filename = file_queue.get(block=False)
        main()

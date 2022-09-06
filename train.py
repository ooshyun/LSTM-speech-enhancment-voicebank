import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import IPython.display as ipd
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
import zipfile
from data_processing.feature_extractor import FeatureExtractor
from utils import prepare_input_features
from model import build_model, build_model_lstm
# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from tensorflow.python.client import device_lib
import keras.models
from pathlib import Path

device_lib.list_local_devices()

tf.random.set_seed(999)
np.random.seed(999)

# model_name = 'cnn'
model_name = 'lstm'

path_to_dataset = f"./records_{model_name}"

# get training and validation tf record file names
train_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'train_*'))
val_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'val_*'))

# suffle the file names for training
np.random.shuffle(train_tfrecords_filenames)
print("Training file names: ", train_tfrecords_filenames)
print("Validation file names: ", val_tfrecords_filenames)

if model_name == "cnn":
    windowLength = 256
    overlap      = round(0.25 * windowLength) # overlap of 75%
    ffTLength    = windowLength
    inputFs      = 48e3
    fs           = 16e3
    numFeatures  = ffTLength//2 + 1
    numSegments  = 8

if model_name == "lstm":
    windowLength = 512
    overlap      = round(0.5 * windowLength) # overlap of 75%
    ffTLength    = windowLength
    inputFs      = 48e3
    fs           = 16e3
    numFeatures  = ffTLength//2 + 1
    numSegments  = 63 # 1 sec in 512 window, 256 hop, sr = 16000 Hz

print("windowLength:",windowLength)
print("overlap:",overlap)
print("ffTLength:",ffTLength)
print("inputFs:",inputFs)
print("fs:",fs)
print("numFeatures:",numFeatures)
print("numSegments:",numSegments)

def tf_record_parser(record):
    keys_to_features = {
        "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
        "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
    clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
    noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)

    if model_name == "cnn":
        # reshape input and annotation images, cnn
        noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (numFeatures, numSegments, 1), name="noise_stft_mag_features")
        clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (numFeatures, 1, 1), name="clean_stft_magnitude")
        noise_stft_phase = tf.reshape(noise_stft_phase, (numFeatures,), name="noise_stft_phase")
    elif model_name == 'lstm':
        # reshape input and annotation images, lstm
        noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (1, numSegments, numFeatures), name="noise_stft_mag_features")
        clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (1, numSegments, numFeatures), name="clean_stft_magnitude")
        noise_stft_phase = tf.reshape(noise_stft_phase, (numFeatures,), name="noise_stft_phase")
    else:
        raise ValueError("Model didn't implement...")

    return noise_stft_mag_features, clean_stft_magnitude

train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
train_dataset = train_dataset.map(tf_record_parser)
train_dataset = train_dataset.shuffle(8192)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(512)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


test_dataset = tf.data.TFRecordDataset([val_tfrecords_filenames])
test_dataset = test_dataset.map(tf_record_parser)
test_dataset = test_dataset.repeat(1)
test_dataset = test_dataset.batch(512)

if model_name == "cnn":
    model = build_model(l2_strength=0.0)
elif model_name == "lstm":
    model = build_model_lstm()
else:
    raise ValueError("Model didn't implement...")
model.summary()

# You might need to install the following dependencies: sudo apt install python-pydot python-pydot-ng graphviz
tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

# %tensorboard --logdir logs

baseline_val_loss = model.evaluate(test_dataset)[0]
print(f"Baseline accuracy {baseline_val_loss}")

def l2_norm(vector):
    return np.square(vector)

def SDR(denoised, cleaned, eps=1e-7): # Signal to Distortion Ratio
    a = l2_norm(denoised)
    b = l2_norm(denoised - cleaned)
    a_b = a / b
    return np.mean(10 * np.log10(a_b + eps))

log_folder = Path(f"./logs/{model_name}")
result_folder = Path(f"./result/{model_name}")

if not log_folder.is_dir():
    log_folder.mkdir()
if not result_folder.is_dir():
    result_folder.mkdir()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, baseline=None)
logdir = os.path.join(f"./logs/{model_name}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')

save_path = f'./result/{model_name}/checkpoint' + "/model-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, 
                                                         monitor='val_loss', save_best_only=True)

model.fit(train_dataset,
         steps_per_epoch=600, # you might need to change this
         validation_data=test_dataset,
         epochs=200,
         callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback]
        )

val_loss = model.evaluate(test_dataset)[0]
if val_loss < baseline_val_loss:
  print("New model saved.")
  keras.models.save_model(model, f'.result/{model_name}/model', overwrite=True, include_optimizer=True)
  # model.save('./denoiser_cnn_log_mel_generator.h5')
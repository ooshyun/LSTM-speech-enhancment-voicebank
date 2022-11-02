import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
import os
import numpy as np
import tensorflow as tf

import glob
import numpy as np
from utils import prepare_input_features, stft_tensorflow
# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

tf.random.set_seed(999)
np.random.seed(999)

# model_name = 'cnn'
model_name = 'lstm'

# domain = 'freq'
domain = 'time'

top_db = 80
center = True

if model_name == "cnn":
    n_fft    = 256
    win_length = n_fft
    overlap      = round(0.25 * win_length) # overlap of 75%
    inputFs      = 48e3
    fs           = 16e3
    numFeatures  = n_fft//2 + 1
    numSegments  = 8

elif model_name == "lstm":
    n_fft    = 512
    win_length = n_fft
    overlap      = round(0.5 * win_length) # overlap of 50%
    inputFs      = 48e3
    fs           = 16e3
    numFeatures  = n_fft//2 + 1
    numSegments  = 64 if center else 62 # 1.008 sec in 512 window, 256 hop, sr = 16000 Hz
else:
    NotImplementedError("Only implemented cnn and lstm")

config = {'top_db': top_db,
    'nfft': n_fft,
    'overlap': round(0.5 * win_length),
    'center': center,
    'fs': 16000,
    'audio_max_duration': 1.008,
    'numFeatures':numFeatures,
    'numSegments':numSegments,
    }

print("-"*30)
for key, value in config.items():
    print(f"{key} : {value}")
print("-"*30)

if model_name == 'lstm':
    if domain == 'time':
        path_to_dataset = f"./records_{model_name}_{domain}"
    else:    
        path_to_dataset = f"./records_{model_name}"
else:
    path_to_dataset = f"./records_{model_name}"

# get training and validation tf record file names
train_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'train_*'))
val_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'val_*'))

# suffle the file names for training
print("Data directory: ", path_to_dataset)
print("Domain: ", domain)
print("Model type: ", model_name)
np.random.shuffle(train_tfrecords_filenames)
print("Training file names: ", train_tfrecords_filenames)
print("Validation file names: ", val_tfrecords_filenames)


def tf_record_parser(record):
    if model_name == "cnn":
        keys_to_features = {
        "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
        "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
        }
        features = tf.io.parse_single_example(record, keys_to_features)

        noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
        clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
        noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)
        # when getting data from tfrecords, it need to transfer tensorflow api such as reshape

        # reshape input and annotation images, cnn
        noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (numFeatures, numSegments, 1), name="noise_stft_mag_features")
        clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (numFeatures, 1, 1), name="clean_stft_magnitude") # [TODO] Check
        noise_stft_phase = tf.reshape(noise_stft_phase, (numFeatures,), name="noise_stft_phase")
    
        noisy_stft = tf.stack([noise_stft_mag_features, noise_stft_phase])
        clean_stft = tf.stack([clean_stft_magnitude, clean_stft_phase])

        return noise_stft_mag_features , clean_stft_magnitude
    
    elif model_name == 'lstm':
        if domain == 'time':    
            keys_to_features = {
            "noisy": tf.io.FixedLenFeature((), tf.string, default_value=""),
            'clean': tf.io.FixedLenFeature([], tf.string),
            }

            features = tf.io.parse_single_example(record, keys_to_features)

            noisy = tf.io.decode_raw(features['noisy'], tf.float32)
            clean = tf.io.decode_raw(features['clean'], tf.float32)

            noise_stft_magnitude, noise_stft_phase, noise_stft_real, noise_stft_imag = stft_tensorflow(noisy, 
                                                                                                    nfft=config['nfft'], 
                                                                                                    hop_length=config['overlap'],
                                                                                                    center=config['center'])
            clean_stft_magnitude, clean_stft_phase, clean_stft_real, clean_stft_imag = stft_tensorflow(clean, 
                                                                                                    nfft=config['nfft'], 
                                                                                                    hop_length=config['overlap'],
                                                                                                    center=config['center'])
        else:          
            keys_to_features = {
                "noisy_stft_magnitude": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string),     
                "noise_stft_phase": tf.io.FixedLenFeature((), tf.string),
                "clean_stft_phase": tf.io.FixedLenFeature((), tf.string),
            }
            features = tf.io.parse_single_example(record, keys_to_features)

            noise_stft_magnitude = tf.io.decode_raw(features['noisy_stft_magnitude'], tf.float32) # phase scaling by clean wav
            clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
            noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)
            clean_stft_phase = tf.io.decode_raw(features['clean_stft_phase'], tf.float32)
            
        noise_stft_magnitude = tf.reshape(noise_stft_magnitude/((numFeatures-1)*2), (1, numSegments, numFeatures), name="noise_stft_magnitude")
        clean_stft_magnitude = tf.reshape(clean_stft_magnitude/((numFeatures-1)*2), (1, numSegments, numFeatures), name="clean_stft_magnitude")
        noise_stft_phase = tf.reshape(noise_stft_phase, (1, numSegments, numFeatures), name="noise_stft_phase")
        clean_stft_phase = tf.reshape(clean_stft_phase, (1, numSegments, numFeatures), name="clean_stft_phase")
    
        noisy_feature = tf.stack([noise_stft_magnitude, noise_stft_phase], name="noisy")
        clean_feature = tf.stack([clean_stft_magnitude, clean_stft_phase], name="clean")

        return noisy_feature, clean_feature
    else:
        raise ValueError("Model didn't implement...")

train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
train_dataset = train_dataset.map(tf_record_parser)
train_dataset = train_dataset.shuffle(128)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for step, batch in enumerate(train_dataset):
    print(step, len(batch), batch[0].shape)
    break
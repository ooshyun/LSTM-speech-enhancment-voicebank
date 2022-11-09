import os
from .utils import stft_tensorflow
import numpy as np
import glob
import tensorflow as tf

def load_dataset(args):
    model_name = args.model.name
    domain = args.dset.domain
    nfft = args.dset.n_fft
    hop_length = args.dset.hop_length
    center = args.dset.center
    num_features = args.dset.n_feature
    num_segments = args.dset.n_segment

    # 2. Load data
    if args.dset.top_db > args.dset.max_db: # 16bit
        path_to_dataset = f"{args.dset.save_path}/records_{args.model.name}_{args.dset.domain}"
    else:
        path_to_dataset = f"{args.dset.save_path}/records_{args.model.name}_{args.dset.domain}_{args.dset.top_db}topdb"
    
    # get training and validation tf record file names
    train_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'train_*'))
    val_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'val_*'))

    # shuffle the file names for training
    np.random.shuffle(train_tfrecords_filenames)
    print("Data path: ", path_to_dataset)
    print("Training file names: ", train_tfrecords_filenames)
    print("Validation file names: ", val_tfrecords_filenames)

    def tf_record_parser(record):
        if model_name == "cnn":
            if domain == 'freq': 
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
                noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (num_features, num_segments, 1), name="noise_stft_mag_features")
                clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (num_features, 1, 1), name="clean_stft_magnitude") # [TODO] Check
                noise_stft_phase = tf.reshape(noise_stft_phase, (num_features,), name="noise_stft_phase")

                return noise_stft_mag_features , clean_stft_magnitude
            elif domain == 'time':
                NotImplementedError("cnn model is not implemented for time domain")
            else:
                NotImplementedError("dataset domain is incorrect")
                        
        elif model_name == 'lstm':
            if domain == 'freq':          
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
            elif domain == 'time':    
                keys_to_features = {
                "noisy": tf.io.FixedLenFeature((), tf.string, default_value=""),
                'clean': tf.io.FixedLenFeature([], tf.string),
                }

                features = tf.io.parse_single_example(record, keys_to_features)

                noisy = tf.io.decode_raw(features['noisy'], tf.float32)
                clean = tf.io.decode_raw(features['clean'], tf.float32)

                noise_stft_magnitude, noise_stft_phase, noise_stft_real, noise_stft_imag = stft_tensorflow(noisy, 
                                                                                                        nfft=nfft, 
                                                                                                        hop_length=hop_length,
                                                                                                        center=center)
                clean_stft_magnitude, clean_stft_phase, clean_stft_real, clean_stft_imag = stft_tensorflow(clean, 
                                                                                                        nfft=nfft, 
                                                                                                        hop_length=hop_length,
                                                                                                        center=center)
            else:
                NotImplementedError("dataset domain is incorrect")

            def scaling(x, normalize):
                if normalize:
                    scale_factor = (num_features-1)*2
                else:
                    scale_factor = 1
                return x/scale_factor

            noise_stft_magnitude = tf.reshape(scaling(noise_stft_magnitude, args.dset.fft_normalize), (1, num_segments, num_features), name="noise_stft_magnitude")
            clean_stft_magnitude = tf.reshape(scaling(clean_stft_magnitude, args.dset.fft_normalize), (1, num_segments, num_features), name="clean_stft_magnitude")
            noise_stft_phase = tf.reshape(noise_stft_phase, (1, num_segments, num_features), name="noise_stft_phase")
            clean_stft_phase = tf.reshape(clean_stft_phase, (1, num_segments, num_features), name="clean_stft_phase")

            noisy_feature = tf.stack([noise_stft_magnitude, noise_stft_phase], name="noisy")
            clean_feature = tf.stack([clean_stft_magnitude, clean_stft_phase], name="clean")

            return noisy_feature, clean_feature
        else:
            raise ValueError("Model didn't implement...")

    train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
    train_dataset = train_dataset.map(tf_record_parser)
    train_dataset = train_dataset.shuffle(8192)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # val_dataset
    test_dataset = tf.data.TFRecordDataset([val_tfrecords_filenames])
    test_dataset = test_dataset.map(tf_record_parser)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(args.batch_size)
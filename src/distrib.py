import os
import glob
from pathlib import Path
from datetime import datetime
import numpy as np

import tensorflow as tf
import keras.callbacks
import keras.models
from .utils import save_json, stft_tensorflow


def save_model_all(path, model: keras.models.Model):
    model_save_path = os.path.join(path, "model")
    optimizer_save_path = os.path.join(path, "optimizer")

    keras.models.save_model(
        model, model_save_path, overwrite=True, include_optimizer=True
    )
    optimizer_save_path = Path(optimizer_save_path)
    if not optimizer_save_path.is_dir():
        optimizer_save_path.mkdir()
    optimizer_save_path = optimizer_save_path / "optim.json"
    save_json({"optimizer": model.optimizer.get_weights()}, optimizer_save_path)


def load_callback(path, args):
    checkpoint_save_path = os.path.join(
        path, "checkpoint/checkpoint-{epoch:02d}-{val_loss:.9f}.hdf5"
    )
    console_log_save_path = os.path.join(path, "debug.txt")

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True, baseline=None
    )
    logdir = os.path.join(path, "logs/")
    tensorboard_callback = keras.callbacks.TensorBoard(
        logdir, update_freq="batch", histogram_freq=1, write_graph=True
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path, test="val_loss", save_best_only=True
    )
    # time_callback = TimeHistory(filepath=console_log_save_path)
    # histogram_freq=0, write_graph=True: for monitoring the weight histogram

    return [
        early_stopping_callback,
        tensorboard_callback,
        checkpoint_callback,
        # time_callback,
    ]


def load_model(args):
    model_name = args.model.name

    if model_name in ("rnn", "lstm", "gru"):
        from .model.rnn import build_model_rnn
        model = build_model_rnn(args)
    elif model_name == "crn":
        from .model.crn import build_crn_model_tf
        model = build_crn_model_tf(args)
    elif model_name == "unet":
        from .model.unet import build_unet_model_tf
        model = build_unet_model_tf(args)        
    elif model_name == "conv-tasnet":
        from .model.conv_tasnet import build_conv_tasnet_model_tf
        model = build_conv_tasnet_model_tf(args)
    else:
        raise ValueError("Model didn't implement...")
    model.summary()

    if args.model.path is not None:
        print("Loading Model...")
        if args.model.ckpt:
            model.load_weights(os.path.join(args.model.path, args.model.ckpt))
        else:
            model = keras.models.load_model(
                os.path.join(args.model.path, "model"), compile=False
            )

    if model_name in ("rnn", "lstm", "gru"):
        from .model.rnn import compile_model
    elif model_name == "crn":
        from .model.crn import compile_model
    elif model_name == "unet":
        from .model.unet import compile_model
    elif model_name == "conv-tasnet":
        from .model.conv_tasnet import compile_model

    compile_model(model, args)

    return model


def load_dataset(args):
    model_name = args.model.name
    save_path = args.dset.save_path
    flag_fft = args.dset.fft
    nfft = args.dset.n_fft
    hop_length = args.dset.hop_length
    center = args.dset.center
    sample_rate = args.dset.sample_rate
    segment = args.dset.segment

    num_features = args.model.n_feature
    num_segments = args.model.n_segment
    normalization = args.dset.normalize
    fft_normalization = args.model.fft_normalization
    top_db = args.dset.top_db
    train_split = int(args.dset.split * 100)

    if args.dset.segment_normalization:
        seg_normalization = args.dset.segment_normalization
    else:
        seg_normalization = False

    # 2. Load data
    path_to_dataset = f"{save_path}/records_seg_{str(segment).replace('.', '-')}_train_{train_split}_norm_{normalization}_segNorm_{seg_normalization}_fft_{flag_fft}_topdB_{top_db}"
    if args.debug:
        path_to_dataset = path_to_dataset + "_debug"
    path_to_dataset = Path(path_to_dataset)

    # get training and validation tf record file names
    train_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, "train_*"))
    val_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, "val_*"))

    # shuffle the file names for training
    np.random.shuffle(train_tfrecords_filenames)
    print("Data path: ", path_to_dataset)
    print("Training file names: ", len(train_tfrecords_filenames))
    print("Validation file names: ", len(val_tfrecords_filenames))

    def tf_record_parser(record):
        if model_name in ("unet", "conv-tasnet"):
            if flag_fft:
                raise ValueError(f"{model_name} should flag of fft False in configuration...")

        if flag_fft:
            keys_to_features = {
                "noisy_stft_real": tf.io.FixedLenFeature(
                    [], tf.string, default_value=""
                ),
                "clean_stft_real": tf.io.FixedLenFeature((), tf.string),
                "noisy_stft_imag": tf.io.FixedLenFeature((), tf.string),
                "clean_stft_imag": tf.io.FixedLenFeature((), tf.string),
            }
            features = tf.io.parse_single_example(record, keys_to_features)

            noisy_stft_real = tf.io.decode_raw(
                features["noisy_stft_real"], tf.float32
            )  # phase scaling by clean wav
            clean_stft_real = tf.io.decode_raw(
                features["clean_stft_real"], tf.float32
            )
            noisy_stft_imag = tf.io.decode_raw(
                features["noisy_stft_imag"], tf.float32
            )
            clean_stft_imag = tf.io.decode_raw(
                features["clean_stft_imag"], tf.float32
            )
            
            noisy_feature = tf.complex(real=noisy_stft_real, imag=noisy_stft_imag)
            clean_feature = tf.complex(real=clean_stft_real, imag=clean_stft_imag)

            if fft_normalization:
                noisy_feature = tf.divide(noisy_feature, nfft)
                clean_feature = tf.divide(clean_feature, nfft)
            
        else:
            keys_to_features = {
                "noisy": tf.io.FixedLenFeature((), tf.string, default_value=""),
                "clean": tf.io.FixedLenFeature((), tf.string),
            }

            features = tf.io.parse_single_example(record, keys_to_features)

            noisy = tf.io.decode_raw(features["noisy"], tf.float32)
            clean = tf.io.decode_raw(features["clean"], tf.float32)


            noisy_feature = stft_tensorflow(wav=noisy, 
                                            nfft=nfft, 
                                            hop_length=hop_length, 
                                            center=center, 
                                            normalize=fft_normalization
                                            )
            
            clean_feature = stft_tensorflow(wav=clean, 
                                            nfft=nfft, 
                                            hop_length=hop_length, 
                                            center=center, 
                                            normalize=fft_normalization
                                            )
        if model_name in ("unet", "conv-tasnet"):
            noisy_feature = tf.reshape(noisy, (1, int(sample_rate*segment)), name="noisy_feature")
            clean_feature = tf.reshape(clean, (1, int(sample_rate*segment)), name="clean_feature")
        else:
            noisy_feature = tf.reshape(noisy_feature, (1, num_segments, num_features), name="noisy_feature")
            clean_feature = tf.reshape(clean_feature, (1, num_segments, num_features), name="clean_feature")

        return noisy_feature, clean_feature

    """
    TFRecordDataset
      map: map between each filenames and custom function
      shuffle: shuffle the number of data, it can set a seed
      repeat: if repeat 2, then [1, 2] -> [1, 2, 1, 2]
      batch: same as batch concept
      prefetch: prepare while training, if set the buffer size as tf.data.experimental.AUTOTUNE, it use automatic method in keras
    """
    train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
    train_dataset = train_dataset.map(tf_record_parser)
    train_dataset = train_dataset.shuffle(8192)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # DataLossError (see above for traceback): corrupted record at 1261886956
    # node IteratorGetNext
    # Shape/_6
    # https://github.com/tensorflow/tensorflow/issues/13463
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())

    # val_dataset
    test_dataset = tf.data.TFRecordDataset([val_tfrecords_filenames])
    test_dataset = test_dataset.map(tf_record_parser)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(args.batch_size)
    test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())

    return train_dataset, test_dataset

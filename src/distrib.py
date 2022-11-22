import os
import glob
from pathlib import Path
from datetime import datetime
import numpy as np

import tensorflow as tf
import keras.callbacks
import keras.models

from .utils import save_json, stft_tensorflow, TimeHistory


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
    time_callback = TimeHistory(filepath=console_log_save_path)
    # histogram_freq=0, write_graph=True: for monitoring the weight histogram

    return [
        early_stopping_callback,
        tensorboard_callback,
        checkpoint_callback,
        time_callback,
    ]


def load_model(args):
    model_name = args.model.name

    if model_name == "lstm":
        from .lstm import build_model_lstm

        model = build_model_lstm(args)
    else:
        raise ValueError("Model didn't implement...")
    model.summary()

    if args.model.path is not None:
        if args.model.ckpt:
            model.load_weights(os.path.join(args.model.path, args.model.ckpt))
        else:
            model = keras.models.load_model(
                os.path.join(args.model.path, "model"), compile=False
            )

    if model_name == "lstm":
        from .lstm import compile_model

        compile_model(model, args)

    return model


def load_dataset(args):
    model_name = args.model.name
    save_path = args.dset.save_path
    flag_fft = args.dset.fft
    nfft = args.dset.n_fft
    hop_length = args.dset.hop_length
    center = args.dset.center
    num_features = args.model.n_feature
    num_segments = args.model.n_segment
    normalization = args.dset.normalize
    fft_normalization = args.dset.fft_normalize
    top_db = args.dset.top_db
    train_split = int(args.dset.split * 100)

    if args.dset.segment_normalization:
        seg_normalization = args.dset.segment_normalization
    else:
        seg_normalization = False

    # 2. Load data
    path_to_dataset = f"{save_path}/records_{model_name}_train_{train_split}_norm_{normalization}_segNorm_{seg_normalization}_fft_{flag_fft}"
    if fft_normalization:
        path_to_dataset = path_to_dataset + f"_norm_topdB_{top_db}"
    else:
        path_to_dataset = path_to_dataset + f"_topdB_{top_db}"
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
        if model_name == "lstm":
            if flag_fft:
                keys_to_features = {
                    "noisy_stft_magnitude": tf.io.FixedLenFeature(
                        [], tf.string, default_value=""
                    ),
                    "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string),
                    "noise_stft_phase": tf.io.FixedLenFeature((), tf.string),
                    "clean_stft_phase": tf.io.FixedLenFeature((), tf.string),
                }
                features = tf.io.parse_single_example(record, keys_to_features)

                noise_stft_magnitude = tf.io.decode_raw(
                    features["noisy_stft_magnitude"], tf.float32
                )  # phase scaling by clean wav
                clean_stft_magnitude = tf.io.decode_raw(
                    features["clean_stft_magnitude"], tf.float32
                )
                noise_stft_phase = tf.io.decode_raw(
                    features["noise_stft_phase"], tf.float32
                )
                clean_stft_phase = tf.io.decode_raw(
                    features["clean_stft_phase"], tf.float32
                )
            else:
                keys_to_features = {
                    "noisy": tf.io.FixedLenFeature((), tf.string, default_value=""),
                    "clean": tf.io.FixedLenFeature([], tf.string),
                }

                features = tf.io.parse_single_example(record, keys_to_features)

                noisy = tf.io.decode_raw(features["noisy"], tf.float32)
                clean = tf.io.decode_raw(features["clean"], tf.float32)

                (
                    noise_stft_magnitude,
                    noise_stft_phase,
                    noise_stft_real,
                    noise_stft_imag,
                ) = stft_tensorflow(
                    noisy, nfft=nfft, hop_length=hop_length, center=center
                )
                (
                    clean_stft_magnitude,
                    clean_stft_phase,
                    clean_stft_real,
                    clean_stft_imag,
                ) = stft_tensorflow(
                    clean, nfft=nfft, hop_length=hop_length, center=center
                )

            noise_stft_magnitude = tf.reshape(
                noise_stft_magnitude,
                (1, num_segments, num_features),
                name="noise_stft_magnitude",
            )
            clean_stft_magnitude = tf.reshape(
                clean_stft_magnitude,
                (1, num_segments, num_features),
                name="clean_stft_magnitude",
            )
            noise_stft_phase = tf.reshape(
                noise_stft_phase,
                (1, num_segments, num_features),
                name="noise_stft_phase",
            )
            clean_stft_phase = tf.reshape(
                clean_stft_phase,
                (1, num_segments, num_features),
                name="clean_stft_phase",
            )

            noisy_feature = tf.stack(
                [noise_stft_magnitude, noise_stft_phase], name="noisy"
            )
            clean_feature = tf.stack(
                [clean_stft_magnitude, clean_stft_phase], name="clean"
            )

            return noisy_feature, clean_feature

        else:
            raise ValueError("Model didn't implement...")

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

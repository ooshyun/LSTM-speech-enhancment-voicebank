"""
Recurrent type Network, RNN, LSTM and GRU
"""

from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    SimpleRNN,
    LSTM,
    GRU,
    Layer,
    Multiply,

)

import keras.layers
import tensorflow as tf
from keras import Model
import keras.regularizers
import keras.optimizers

import os
import numpy as np
from src.utils import load_json

from .metrics import (
    CustomMetric,
    SpeechMetric,
)
from .loss import (
    mean_square_error_amplitdue_phase,
    mean_absolute_error_amplitdue_phase,
    ideal_amplitude_mask,
    phase_sensitive_spectral_approximation_loss,
    phase_sensitive_spectral_approximation_loss_bose,
)

from .time_frequency import(
    Magnitude,
    MelSpec,
    InverseMelSpec,
    ContextLayer,
    DeContextLayer,
)


def build_model_rnn(args):
    inputs = Input(
        shape=[1, int(args.dset.segment*args.dset.sample_rate//args.dset.hop_length + 1), args.model.n_feature],
        name="input", 
        dtype=tf.complex64,
    )
    
    mask = Magnitude()(inputs)

    # print(mask.shape, mask.dtype)

    mask = tf.squeeze(mask, axis=1)  # merge channel
    
    # print(mask.shape, mask.dtype)

    mask = MelSpec(args)(mask)
    mask = ContextLayer(unit=args.model.n_mels, use_bias=True)(mask)

    # print(mask.shape, mask.dtype)
    
    if args.model.name == 'rnn':
        mask = SimpleRNN(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
        mask = SimpleRNN(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
    if args.model.name == 'lstm':
        mask = LSTM(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
        mask = LSTM(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
    if args.model.name == 'gru':
        mask = GRU(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
        mask = GRU(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
    
    mask = BatchNormalization()(mask)

    mask = Dense(
        args.model.n_mels,
        activation="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(
        mask
    )  
    mask = Dense(
        args.model.n_mels,
        activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(
        mask
    )  

    # print(mask.shape, mask.dtype)
    mask = DeContextLayer()(mask)
    mask = InverseMelSpec(args)(mask)

    # print(mask.shape, mask.dtype)

    mask = tf.expand_dims(mask, axis=1)  # expand channel
    mask = tf.cast(mask, dtype=tf.complex64)
    # print(mask.shape, mask.dtype)

    outputs = Multiply()(
        [inputs, mask]
    )

    # print(outputs.shape, outputs.dtype)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def compile_model(model: Model, args):
    """check baseline
        model.compile(optimizer=optimizer,
                    loss= meanSquareError(), # 'mse'
                    metrics=[keras.metrics.RootMeanSquaredError('rmse'),
                    ])
    """
    if args.optim.optim == "adam":
        optimizer = keras.optimizers.Adam(args.optim.lr)
    elif args.optim.optim == "sgd":
        optimizer = keras.optimizers.SGD(args.optim.lr)
    else:
        raise NotImplementedError(f"Optimizer {args.optim.optim} is not implemented")

    if args.optim.loss == "mse":
        loss_function = mean_square_error_amplitdue_phase
    elif args.optim.loss == "rmse":
        loss_function = mean_absolute_error_amplitdue_phase
    elif args.optim.loss == "ideal-mag":
        loss_function = ideal_amplitude_mask
    elif args.optim.loss == "psa":
        loss_function = phase_sensitive_spectral_approximation_loss
    elif args.optim.loss == "psa-bose":
        loss_function = phase_sensitive_spectral_approximation_loss_bose
    else:
        raise NotImplementedError(f"Loss '{self.metric}' is not implemented")

    if args.model.path is not None and args.optim.load:
        if "optimizer" in os.listdir(args.model.path):  # optimizer folder check
            tf.print("Optimizer Loading...")
            dummpy_model = build_model_rnn(args)
            optimizer_state = load_json(
                os.path.join(args.model.path, "optimizer/optim.json")
            )["optimizer"]
            dummy_batch_size = 1
            dummy_tensor = tf.ones(
                shape=(
                    dummy_batch_size,
                    1,
                    int(args.dset.segment*args.dset.sample_rate//args.dset.hop_length + 1),
                    args.model.n_feature,
                ),
                dtype=tf.complex64,
            )
            dummy_clean_tensor = tf.ones(
                shape=(
                    dummy_batch_size,
                    1,
                    int(args.dset.segment*args.dset.sample_rate//args.dset.hop_length + 1),
                    args.model.n_feature,
                ), 
                dtype=tf.complex64,
            )

            dummpy_model.compile(
                optimizer=optimizer,
                loss=loss_function,
            )
            dummpy_model.fit(
                x=dummy_tensor, y=dummy_clean_tensor, batch_size=dummy_batch_size
            )
            
            del (
                dummpy_model,
                dummy_tensor,
                dummy_clean_tensor,
            )  # [TODO] How to remove object and check it removed?
            
            # Currently, when using saved optimizer, the next optimizer loading has almost double size
            # The reason to load dummy model is when optimizer didn't compile, then it didn't have the optimizer weight
            # Ex. the case fo LSTM model: 25 -> 49, so it temporaily saved half of len
            # This tested in test_model and optimizer/model weight is same between saving and loading
            # [TODO] How to load optimizer weights? 
            len_optimizer = len(optimizer.get_weights()) 
            optimizer.set_weights(optimizer_state[:len_optimizer])

            tf.print("Optimizer was loaded!")
        else:
            tf.print("Optimizer was not existed!")

    metrics = [
        SpeechMetric(
            model_name = args.model.name,
            n_fft=args.dset.n_fft,
            hop_length=args.dset.hop_length,
            normalize=args.model.fft_normalization,
            name=metric_name,
        )
        for metric_name in args.model.metric
    ]
    metrics.append(CustomMetric(metric=args.optim.loss, name=args.optim.loss))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

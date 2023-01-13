"""
Convolutional Recurrent Network, Conversion to Tensorflow

Reference
---------
https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement/
"""

import tensorflow as tf
import keras
import keras.layers

import os
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

import tensorflow as tf
import numpy as np
import keras.layers
from .time_frequency import(
    Magnitude,
    SqueezeChannel,
)

class ZeroPadding(keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)        
        self.paddings = padding
    
        
    def call(self, inputs, training=True):
        paddings = np.zeros(shape=(len(inputs.shape), 2), dtype=np.int32)
        for ipad in range(len(self.paddings)):
            for loc in range(len(self.paddings[0])):
                paddings[len(inputs.shape)-len(self.paddings)+ipad, loc] = self.paddings[ipad][loc]
        paddings = tf.constant(paddings)

        outputs = tf.pad(inputs, paddings, "CONSTANT")
        return outputs

    def get_config(self):
        config = super(ZeroPadding, self).get_config()
        config.update(
            {
                "padding": self.paddings,
            }
        )
        return config

class CausalConvBlockTF(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.pad = ZeroPadding(padding=((1, 1), (0, 0), (0, 0)))
        self.conv = keras.layers.Conv2D(
            filters = out_channels,
            kernel_size = (2, 3),
            strides = (1, 2),
            padding= "valid",
        )
        self.norm = keras.layers.BatchNormalization()
        self.activation = keras.layers.ELU()

    def call(self, inputs):
        """
        2D Causal convolution.
        Args:
            x: [B, T, F, C]

        Returns:
            [B, T, F, C]
        """
        if self.in_channels == 1:
            outputs = tf.expand_dims(inputs, axis=-1)
        else:
            outputs = inputs
        outputs = self.pad(outputs)
        outputs = self.conv(outputs)
        outputs = outputs[:, :-1, ...]
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(CausalConvBlockTF, self).get_config()
        config.update(
            {
                "in_channels": self.in_channels,
            }
        )
        return config

class CausalTransConvBlockTF(keras.layers.Layer):
    def __init__(self, out_channels, is_last=False, output_padding=None):
        super().__init__()
        self.out_channels = out_channels
        self.is_last = is_last
        self.output_padding = output_padding


        self.conv = keras.layers.Conv2DTranspose(
            filters=out_channels,
            kernel_size=(2, 3),
            strides=(1, 2),
            output_padding = self.output_padding,
            padding="valid",
        )
            
        self.norm = keras.layers.BatchNormalization()
        if self.is_last:
            self.activation = keras.layers.ReLU()
        else:
            self.activation = keras.layers.ELU()

    def call(self, inputs):
        """
        2D Causal convolution.
        Args:
            x: [B, T, F, C]

        Returns:
            [B, T, F, C]
        """
        outputs = inputs
        outputs = self.conv(outputs)
        outputs = outputs[:, :-1, ...] 
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        if self.out_channels == 1:
            outputs = tf.squeeze(outputs, axis=-1)
        return outputs

    def get_config(self):
        config = super(CausalTransConvBlockTF, self).get_config()
        config.update(
            {
                "is_last": self.is_last,
                "output_padding": self.output_padding,
                "out_channels": self.out_channels,
            }
        )
        return config

def build_crn_model_tf(args):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    inputs = keras.layers.Input(
        shape=[1, int(args.dset.segment*args.dset.sample_rate//args.dset.hop_length + 1), args.model.n_feature],
        name="input",
        dtype=tf.complex64,
    )

    # print(inputs.shape, inputs.dtype)

    magnitude = Magnitude()(inputs)

    # print(magnitude.shape, magnitude.dtype)

    assert magnitude.dtype == tf.float32

    mask = SqueezeChannel()(magnitude)

    # Encoder
    conv_block_1 = CausalConvBlockTF(in_channels=1, out_channels=16)(mask)
    conv_block_2 = CausalConvBlockTF(in_channels=16, out_channels=32)(conv_block_1)
    conv_block_3 = CausalConvBlockTF(in_channels=32, out_channels=64)(conv_block_2)
    conv_block_4 = CausalConvBlockTF(in_channels=64, out_channels=128)(conv_block_3)
    conv_block_5 = CausalConvBlockTF(in_channels=128, out_channels=256)(conv_block_4)

    _, n_frame_size, n_channels, n_f_bins = conv_block_5.shape

    # LSTM
    reshape_1 = keras.layers.Reshape(target_shape=(n_frame_size, n_channels * n_f_bins))(conv_block_5)
    lstm_layer_1 = keras.layers.LSTM(units=n_channels * n_f_bins, 
                                        activation='tanh', 
                                        return_sequences=True)(reshape_1)
    lstm_layer_2 = keras.layers.LSTM(units=n_channels * n_f_bins, 
                                        activation='tanh', 
                                        return_sequences=True)(lstm_layer_1)
    lstm_out = keras.layers.Reshape(target_shape=(n_frame_size, n_channels, n_f_bins))(lstm_layer_2)

    # Decoder
    tran_conv_block_1 = CausalTransConvBlockTF(out_channels=128)(tf.concat((lstm_out, conv_block_5), -1))
    tran_conv_block_2 = CausalTransConvBlockTF(out_channels=64)(tf.concat((tran_conv_block_1, conv_block_4), -1))
    tran_conv_block_3 = CausalTransConvBlockTF(out_channels=32)(tf.concat((tran_conv_block_2, conv_block_3), -1))
    tran_conv_block_4 = CausalTransConvBlockTF(out_channels=16, output_padding=(0, 1))(tf.concat((tran_conv_block_3, conv_block_2), -1))
    tran_conv_block_5 = CausalTransConvBlockTF(out_channels=1, is_last=True)(tf.concat((tran_conv_block_4, conv_block_1), -1))
    
    mask = tf.expand_dims(tran_conv_block_5, axis=1)  # expand channel
    mask = tf.cast(mask, dtype=tf.complex64)

    outputs = keras.layers.Multiply()(
        [inputs, mask]
    )

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def compile_model(model: keras.Model, args):
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
            dummpy_model = build_crn_model_tf(args)
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
            )  
            
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

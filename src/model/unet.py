"""
Wave-U-Net, Conversion to Tensorflow

Refernece. https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement
"""
import numpy as np
import tensorflow as tf
import keras
import keras.layers
from src.utils import load_json

import os

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

class Resample1DInterpolation(keras.layers.Layer):
    def __init__(self, scale_factor, mode, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

        if mode == 'linear':
            self.mode = 'bilinear'
        elif mode == 'nearset':
            self.mode = 'nearest'
        elif mode == 'cubic':
            self.mode = 'bicubic'
        else:
            raise NotImplementedError
            
    def call(self, inputs, training=True):        
        assert len(inputs.shape) == 3 

        _, _, features = inputs.shape # batch, channel, features
        outputs = tf.transpose(inputs, perm=(0, 2, 1))
        outputs = tf.expand_dims(outputs, axis=1)
        outputs = keras.layers.Resizing(height=1, width=features*self.scale_factor, interpolation=self.mode)(outputs)
        outputs = tf.squeeze(outputs, axis=1)
        outputs = tf.transpose(outputs, perm=(0, 2, 1))

        return outputs
    
    def get_config(self):
        config = super(Resample1DInterpolation, self).get_config()
        config.update(
            {
                "scale_factor": self.scale_factor,
                "mode": self.mode,
            }
        )
        return config

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

class DownSamplingLayerTF(keras.layers.Layer):
    def __init__(self, channel_in, channel_out, kernel_size=15, stride=1, padding=7, dilation=1):
        super(DownSamplingLayerTF, self).__init__()
        
        self.main = keras.Sequential([
            ZeroPadding(padding=((padding, padding), )),
            keras.layers.Conv1D(filters=channel_out, 
                                kernel_size=kernel_size,
                                strides=stride,
                                dilation_rate=dilation,
                                data_format='channels_first',
                                ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ])
        
    def call(self, inputs):
        outputs = self.main(inputs)
        return outputs

class UpSamplingLayerTF(keras.layers.Layer):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayerTF, self).__init__()
        
        self.main = keras.Sequential([
            ZeroPadding(padding=((padding, padding), )),
            keras.layers.Conv1D(filters=channel_out, 
                                kernel_size=kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ])

    def call(self, inputs):
        outputs = self.main(inputs)
        return outputs

class Unet(keras.layers.Layer):
    def __init__(self, n_layers=12, channels_interval=24):
        super(Unet, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        tf.print("Unet encoder channel")
        tf.print("In : ", encoder_in_channels_list)
        tf.print("OUt: ", encoder_out_channels_list)

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = []
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayerTF(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                )
            )

        self.middle = keras.Sequential([
            ZeroPadding(padding=((7, 7), )),
            keras.layers.Conv1D(filters=self.n_layers * self.channels_interval, 
                                kernel_size=15, 
                                strides=1,
                                data_format='channels_first',
                                ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ])

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = []
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayerTF(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = keras.Sequential([
            keras.layers.Conv1D(filters=1, 
                                kernel_size=1, strides=1,
                                data_format='channels_first',
                                ),
            keras.layers.Activation('tanh'),
        ])

    def call(self, input):
        encode_block = []
        o = input

        # Down Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            encode_block.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = Resample1DInterpolation(scale_factor=2, mode="linear")(o)
            # Skip Connection, channel concat
            o = tf.concat([o, encode_block[self.n_layers - i - 1]], axis=-2) 
            o = self.decoder[i](o)

        o = tf.concat([o, input], axis=-2) 
        o = self.out(o)
        return o

    def get_config(self):
        config = super(Unet, self).get_config()
        config.update(
            {
                "n_layers": self.n_layers,
                "channels_interval": self.channels_interval,
            }
        )
        return config

def build_unet_model_tf(args):

    inputs = keras.layers.Input(
        shape=[1, int(args.dset.sample_rate*args.dset.segment)],
        name="input",
        dtype=tf.float32,
    )

    outputs = Unet()(inputs)

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
            dummpy_model = build_unet_model_tf(args)
            optimizer_state = load_json(
                os.path.join(args.model.path, "optimizer/optim.json")
            )["optimizer"]
            dummy_batch_size = 1
            dummy_tensor = tf.ones(
                shape=(
                    dummy_batch_size,
                    1,
                    int(args.dset.sample_rate*args.dset.segment),
                ),
                dtype=tf.float32,
            )
            dummy_clean_tensor = tf.ones(
                shape=(
                    dummy_batch_size,
                    1,
                    int(args.dset.sample_rate*args.dset.segment),
                ), 
                dtype=tf.float32,
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

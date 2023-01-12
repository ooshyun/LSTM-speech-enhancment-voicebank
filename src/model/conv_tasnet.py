"""
Conv-TasNet, Conversion to Tensorflow

Refernece 
----------
- A PyTorch implementation of Conv-TasNet, https://github.com/kaituoxu/Conv-TasNet
- TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation (https://arxiv.org/abs/1809.07454)
"""
import numpy as np
import tensorflow as tf
import keras
import keras.layers
from keras.backend import epsilon

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

class ConvTasNetTF(keras.layers.Layer):
    def __init__(self, N, L, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', **kwarg):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(ConvTasNetTF, self).__init__(**kwarg)
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        # Components
        self.encoder = EncoderTF(L, N)
        self.separator = TemporalConvNetTF(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        self.decoder = DecoderTF(N, L)
        
        # init
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_normal_(p)

    def call(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.get_shape()[-1]
        T_conv = est_source.get_shape()[-1]
        
        est_source = tf.pad(est_source, ((0, 0), (0, 0), (0, T_origin - T_conv)))
        return est_source

    def get_config(self):
        config = super(ConvTasNetTF, self).get_config()
        config.update(
            {   
                "N": self.N,
                "L": self.L,
                "B": self.B,
                "H": self.H,
                "P": self.P,
                "X": self.X,
                "R": self.R,
                "C": self.C,
                "norm_type": self.norm_type,
                "casual": self.causal,
                "mask_nonlinear": self.mask_nonlinear,
            }
        )
        return config

class EncoderTF(keras.layers.Layer):
    def __init__(self, L, N, **kwargs):
        super().__init__(**kwargs)
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = keras.layers.Conv1D(filters=N, 
                                            kernel_size=L, 
                                            strides=L//2, 
                                            use_bias=False,
                                            data_format='channels_first'
                                            )
        self.activation = keras.layers.ReLU()
        
    def call(self, inputs):
        """
        Args:
            inputs: [M, T], M is batch size, T is #samples
        Returns:
            inputs_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        # outputs = tf.expand_dims(inputs, axis=1)  # [M, 1, T]
        outputs = self.conv1d_U(inputs)    # [M, N, K]
        outputs = self.activation(outputs)  # [M, N, K]
        return outputs 

    def get_config(self):
        config = super(EncoderTF, self).get_config()
        config.update(
            {
                "L": self.L,
                "N": self.N,
            }
        )
        return config

class DecoderTF(keras.layers.Layer):
    def __init__(self, N, L, **kwarg):
        super(DecoderTF, self).__init__(**kwarg)
        # Hyper-parameter
        self.N, self.L = N, L
        # Components
        self.linear = keras.layers.Dense(units=L, use_bias=False)

    def call(self, inputs, est_mask):
        """
        Args:
            inputs: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            outputs: [M, C, T]
        """
        # D = W * M
        outputs = tf.expand_dims(inputs, axis=1) * est_mask
        outputs = tf.transpose(outputs, (0, 1, 3, 2))

        # S = DV
        outputs = self.linear(outputs)  # [M, C, K, L]
        outputs = tf.signal.overlap_and_add(outputs, self.L//2)
        return outputs

    def get_config(self):
        config = super(DecoderTF, self).get_config()
        config.update(
            {
                "L": self.L,
                "N": self.N,
            }
        )
        return config

class TemporalConvNetTF(keras.layers.Layer):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', **kwarg):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNetTF, self).__init__(**kwarg)
        # Hyper-parameter
        self.C = int(C)
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNormTF(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = keras.layers.Conv1D(filters=B, 
                                                kernel_size=1,
                                                data_format='channels_first', 
                                                use_bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [TemporalBlockTF(B, H, P, stride=1,
                                         padding=padding,
                                         dilation=dilation,
                                         norm_type=norm_type,
                                         causal=causal)]
            repeats += [keras.Sequential(blocks)]
        temporal_conv_net = keras.Sequential(repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = keras.layers.Conv1D(filters=C*N, 
                                        kernel_size=1, 
                                        data_format='channels_first',
                                        use_bias=False)
        

        # Put together
        self.network = keras.Sequential([
                                        layer_norm,
                                        bottleneck_conv1x1,
                                        temporal_conv_net,
                                        mask_conv1x1,
                                     ])

    def call(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        # [M, N, K] -> [M, C*N, K]
        score = self.network(mixture_w)  

        # [M, C*N, K] -> [M, C, N, K]
        M, N, K = mixture_w.get_shape()
        score = keras.layers.Reshape(target_shape=(self.C, N, K))(score)

        # score = tf.reshape(score, shape=(M, self.C, N, K))
        # score = score.view(M, self.C, N, K) # [M, C*N, K] -> [M, C, N, K]
    
        if self.mask_nonlinear == 'softmax':
            est_mask = keras.layers.Softmax(axis=1)(score)
        elif self.mask_nonlinear == 'relu':
            est_mask = keras.layers.ReLU()(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask

class TemporalBlockTF(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False, **kwarg):
        super(TemporalBlockTF, self).__init__(**kwarg)
        # [M, B, K] -> [M, H, K]
        conv1x1 = keras.layers.Conv1D(filters=out_channels, 
                                    kernel_size=1, 
                                    data_format='channels_first',
                                    use_bias=False)
        prelu = keras.layers.PReLU()
        norm = chose_norm_tf(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConvTF(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        self.net = keras.Sequential([
            conv1x1, 
            prelu, 
            norm, 
            dsconv,
            ])
        
    def call(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)

class DepthwiseSeparableConvTF(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False, **kwarg):
        super(DepthwiseSeparableConvTF, self).__init__(**kwarg)
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        padding_layer = ZeroPadding(padding=((padding, padding), ))
        depthwise_conv = keras.layers.Conv1D(
                                            filters=in_channels,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            data_format='channels_first',
                                            dilation_rate=dilation,
                                            groups=in_channels,
                                            use_bias=False,
                                            )
        if causal:
            chomp = Chomp1dTF(padding)
        activation = keras.layers.PReLU()
        norm = chose_norm_tf(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = keras.layers.Conv1D(
                                            filters=out_channels,
                                            kernel_size=1,
                                            data_format='channels_first',
                                            use_bias=False
                                            )

        # Put together
        if causal:
            self.net = keras.Sequential([
                padding_layer,
                depthwise_conv, 
                chomp,
                activation,
                norm,
                pointwise_conv,
            ])
        else:
            self.net = keras.Sequential([
                padding_layer,
                depthwise_conv, 
                activation,
                norm,
                pointwise_conv,
            ])

    def call(self, inputs):
        """
        Args:
            inputs: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(inputs)

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

class Chomp1dTF(keras.layers.Layer):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size, **kwarg):
        super(Chomp1dTF, self).__init__(**kwarg)
        self.chomp_size = chomp_size

    def call(self, inputs):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return inputs[..., :-self.chomp_size]

    def get_config(self):
        config = super(Chomp1dTF, self).get_config()
        config.update(
            {
                "chomp_size": self.chomp_size,
            }
        )
        return config

def chose_norm_tf(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNormTF(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNormTF(channel_size)
    else: # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return keras.layers.BatchNormalization()

class ChannelwiseLayerNormTF(keras.layers.Layer):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size, **kwarg):
        super(ChannelwiseLayerNormTF, self).__init__(**kwarg)

        initvar = np.ones(shape=(1, channel_size, 1), dtype=np.float32)
        self.gamma = tf.Variable(tf.constant(initvar), trainable=True) # [1, N, 1]
        initvar.fill(0)
        self.beta = tf.Variable(tf.constant(initvar), trainable=True) # [1, N, 1]
        
    def call(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = tf.math.reduce_mean(y, axis=1, keepdims=True)  # [M, 1, K]
        # unbiased -> Bessel correction
        var = tf.math.reduce_variance(y, axis=1, keepdims=True)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / tf.math.pow(var + epsilon(), 0.5) + self.beta
        return cLN_y


class GlobalLayerNormTF(keras.layers.Layer):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size, **kwarg):
        super(GlobalLayerNormTF, self).__init__(**kwarg)

        initvar = np.ones(shape=(1, channel_size, 1), dtype=np.float32)
        self.gamma = tf.Variable(tf.constant(initvar), trainable=True) # [1, N, 1]
        initvar.fill(0)
        self.beta = tf.Variable(tf.constant(initvar), trainable=True) # [1, N, 1]
        
    def call(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        mean = tf.math.reduce_mean(y, axis=1, keepdims=True)  # [M, 1, K]
        mean = tf.math.reduce_mean(mean, axis=2, keepdims=True) # [M, 1, 1]
        # unbiased -> Bessel correction
        var = tf.math.reduce_variance(y, axis=1, keepdims=True)  # [M, 1, K]
        var = tf.math.reduce_variance(y, axis=2, keepdims=True)  # [M, 1, 1]
        gLN_y = self.gamma * (y - mean) / tf.math.pow(var + epsilon(), 0.5) + self.beta
        return gLN_y

def build_conv_tasnet_model_tf(args):
    inputs = keras.layers.Input(
        shape=[1, int(args.dset.sample_rate*args.dset.segment)],
        name="input",
        dtype=tf.float32,
    )

    outputs = ConvTasNetTF(N=256, 
                            L=20, 
                            B=256, 
                            H=512, 
                            P=3, 
                            X=8, 
                            R=4, 
                            C=1, 
                            norm_type='gLN', 
                            causal= False, 
                            mask_nonlinear='relu')(inputs)

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
            dummpy_model = build_conv_tasnet_model_tf(args)
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

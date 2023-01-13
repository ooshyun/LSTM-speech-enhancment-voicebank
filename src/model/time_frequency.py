import tensorflow as tf
from keras.backend import epsilon
import keras.layers

class ExponentialMovingAverage(keras.layers.Layer):
    def __init__(
        self,
        alpha = 0.85,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.delay_buffer = None
        self.ema_parameter = [tf.constant(alpha, dtype=tf.float32), 
                            tf.constant(1-alpha, dtype=tf.float32)]
        
    def call(self, inputs, training=True):
        if self.delay_buffer is None:
            outputs = self.ema_parameter[0]*inputs
            if inputs.shape[0] != None:
                self.delay_buffer = self.ema_parameter[0]*inputs + self.ema_parameter[1]*self.delay_buffer
        else:
            outputs = self.ema_parameter[0]*inputs + self.ema_parameter[1]*self.delay_buffer
            self.delay_buffer = self.ema_parameter[0]*inputs + self.ema_parameter[1]*self.delay_buffer                    
        return outputs

    def get_config(self):
        config = super(ExponentialMovingAverage, self).get_config()
        config.update(
            { 
                "alpha": self.alpha,
            }
        )
        return config

class MelSpec(keras.layers.Layer):
    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = args.dset.n_fft
        self.frame_step = args.dset.hop_length
        self.fft_length = args.dset.n_fft
        self.sampling_rate = args.dset.sample_rate
        self.num_mel_channels = args.model.n_mels
        self.freq_min = args.model.f_min
        self.freq_max = args.model.f_max

        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_channels,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, magnitude, training=True):
        # We will only perform the transformation during training.
        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
        return mel

    def get_config(self):
        config = super(MelSpec, self).get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "num_mel_channels": self.num_mel_channels,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
            }
        )
        return config


class InverseMelSpec(keras.layers.Layer):
    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = args.dset.n_fft
        self.frame_step = args.dset.hop_length
        self.fft_length = args.dset.n_fft
        self.sampling_rate = args.dset.sample_rate
        self.num_mel_channels = args.model.n_mels
        self.freq_min = args.model.f_min
        self.freq_max = args.model.f_max

        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_channels,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, mel, training=True):
        # We will only perform the transformation during training.
        magnitude = tf.matmul(mel, tf.transpose(self.mel_filterbank, perm=[1, 0]))
        return magnitude

    def get_config(self):
        config = super(InverseMelSpec, self).get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "num_mel_channels": self.num_mel_channels,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
            }
        )
        return config

class Magnitude(keras.layers.Layer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
    def call(self, inputs, training=True):
        assert inputs.dtype == tf.complex64
        outputs = tf.math.abs(inputs)
        return outputs

class Phase(keras.layers.Layer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
    def call(self, inputs, training=True):
        assert inputs.dtype == tf.complex64
        outputs = tf.math.angle(inputs)
        return outputs

class SqueezeChannel(keras.layers.Layer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
    def call(self, inputs, training=True):
        if inputs.shape[-3] == 1:
            outputs = inputs[..., 0, :, :]
        else:
            raise NotImplementedError("Mono Channel is only acceptable")
        return outputs
 
class ExpansionChannel(keras.layers.Layer):
    def __init__(
        self,
        channels,
        **kwargs,
    ):
        self.channels = channels
        super().__init__(**kwargs)
        
    def call(self, inputs, training=True):
        if self.channels == 1:
            outputs = tf.expand_dims(inputs, axis=-3)
        else:
            raise NotImplementedError("Mono Channel is only acceptable")
        return outputs

class CombineAmplitudePhase(keras.layers.Layer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def call(self, inputs, training=True):
        amplitude = tf.cast(inputs[0], dtype=tf.complex64) + epsilon()
        angle = tf.complex(tf.zeros_like(inputs[1]), inputs[1])
        outputs = tf.multiply(amplitude, tf.exp(angle))
        return outputs
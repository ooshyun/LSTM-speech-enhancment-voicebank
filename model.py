from curses import window
from keras.layers import (
  Conv2D, 
  Input, 
  LeakyReLU, 
  Flatten, 
  Dense, 
  Reshape, 
  Conv2DTranspose, 
  BatchNormalization, 
  Activation, 
  ZeroPadding2D, 
  SpatialDropout2D,
  LSTM,
  Dense,
  Layer,
  Multiply,
)

import tensorflow as tf
from keras import Model, Sequential
import keras.regularizers
from librosa.filters import mel
from librosa import istft
import logging

import numpy as np
import keras.optimizers


from metrics import (
    SI_SDR,
    WB_PESQ,
    SDR,
    STOI,
    NB_PESQ
)

# _model_name = 'cnn'
_model_name = 'lstm'

if _model_name == "cnn":
    win_length = 256
    overlap      = round(0.25 * win_length) # overlap of 75%
    n_fft    = win_length
    inputFs      = 48e3
    fs           = 16e3
    numFeatures  = n_fft//2 + 1
    numSegments  = 8

elif _model_name == "lstm":
    win_length = 512
    overlap      = round(0.5 * win_length) # overlap of 50%
    n_fft    = win_length
    inputFs      = 48e3
    fs           = 16e3
    numFeatures  = n_fft//2 + 1
    numSegments  = 62 # 1.008 sec in 512 window, 256 hop, sr = 16000 Hz

print("win_length:",win_length)
print("overlap:",overlap)
print("n_fft:",n_fft)
print("inputFs:",inputFs)
print("fs:",fs)
print("numFeatures:",numFeatures)
print("numSegments:",numSegments)

def conv_block(x, filters, kernel_size, strides, padding='same', use_bn=True):
  x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False,
          kernel_regularizer=keras.regularizers.l2(0.0006))(x)
  x = Activation('relu')(x)
  if use_bn:
    x = BatchNormalization()(x)
  return x

def full_pre_activation_block(x, filters, kernel_size, strides, padding='same', use_bn=True):
  shortcut = x
  in_channels = x.shape[-1]

  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)

  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=in_channels, kernel_size=kernel_size, strides=strides, padding='same')(x)

  return shortcut + x


def build_model(l2_strength):
  inputs = Input(shape=[numFeatures, numSegments, 1])
  x = inputs
  
  # -----
  x = ZeroPadding2D(((4,4), (0,0)))(x)
  x = Conv2D(filters=18, kernel_size=[9,8], strides=[1, 1], padding='valid', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  skip0 = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                 kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(skip0)
  x = BatchNormalization()(x)

  x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # -----
  x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  skip1 = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                 kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(skip1)
  x = BatchNormalization()(x)

  x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)
  
  x = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
             kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = x + skip1
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
             kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = x + skip0
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
              kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)

  # ----
  x = SpatialDropout2D(0.2)(x)
  x = Conv2D(filters=1, kernel_size=[129,1], strides=[1, 1], padding='same')(x)

  model = Model(inputs=inputs, outputs=x)

  optimizer = keras.optimizers.Adam(3e-4)
  #optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=3e-4)

  model.compile(optimizer=optimizer, loss='mse', 
                metrics=[keras.metrics.RootMeanSquaredError('rmse')])
  return model


class MelSpec(Layer):
    def __init__(
        self,
        frame_length=n_fft,
        frame_step=overlap,
        fft_length=None,
        sampling_rate=16000,
        num_mel_channels=128,
        freq_min=125,
        freq_max=8000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        
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

class InverseMelSpec(Layer):
    def __init__(
        self,
        frame_length=n_fft,
        frame_step=overlap,
        fft_length=None,
        sampling_rate=16000,
        num_mel_channels=128,
        freq_min=125,
        freq_max=8000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        
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

def get_mel_filter(samplerate, n_fft, n_mels, fmin, fmax):
    mel_basis = mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return tf.convert_to_tensor(mel_basis, dtype=tf.float32)

# TODO add custom losses and metrics
# import keras.losses 
# import keras.metrics
# keras.losses.mean_absolute_error # loss = mean(abs(y_true - y_pred), axis=-1)
# keras.losses.mean_squared_error # loss = mean(square(y_true - y_pred), axis=-1)
# keras.metrics.RootMeanSquaredError # metric = sqrt(mean(square(y_true - y_pred)))

def meanAbsoluteError():
  """
    loss function based on energy, e = root(square(real^2 + imag^2))
    loss = mean(abs(y_true - y_pred), axis=-1)
  """
  def _loss(y_true, y_pred):
    reference_real = y_true[..., 0, :, :, :] # real/imag, ch, frame, freq
    reference_imag = y_true[..., 1, :, :, :]
    estimation_real = y_pred[..., 0, :, :, :]
    estimation_imag = y_pred[..., 1, :, :, :]

    reference_real = tf.cast(reference_real, dtype=tf.complex64)
    reference_imag = tf.cast(reference_imag, dtype=tf.complex64)
    estimation_real = tf.cast(estimation_real, dtype=tf.complex64)
    estimation_imag = tf.cast(estimation_imag, dtype=tf.complex64)

    reference_stft = reference_real + 1j*reference_imag
    estimation_stft = estimation_real + 1j*estimation_imag
    
    reference_stft = tf.math.abs(reference_stft)
    estimation_stft = tf.math.abs(estimation_stft)

    return tf.keras.losses.mean_absolute_error(reference_stft, estimation_stft)
  return _loss

def meanSquareError():
  """
    loss function based on energy, e = root(square(real^2 + imag^2))
    loss = mean(square(y_true - y_pred), axis=-1)
  """
  def _loss(y_true, y_pred):
    reference_real = y_true[..., 0, :, :, :] # real/imag, ch, frame, freq
    reference_imag = y_true[..., 1, :, :, :]
    estimation_real = y_pred[..., 0, :, :, :]
    estimation_imag = y_pred[..., 1, :, :, :]

    reference_real = tf.cast(reference_real, dtype=tf.complex64)
    reference_imag = tf.cast(reference_imag, dtype=tf.complex64)
    estimation_real = tf.cast(estimation_real, dtype=tf.complex64)
    estimation_imag = tf.cast(estimation_imag, dtype=tf.complex64)

    reference_stft = reference_real + 1j*reference_imag
    estimation_stft = estimation_real + 1j*estimation_imag
    
    # reference_stft = tf.math.abs(tf.math.sqrt(tf.math.pow(reference_real, 2) - tf.math.pow(reference_imag, 2)))
    # estimation_stft = tf.math.abs(tf.math.sqrt(tf.math.pow(estimation_real, 2) - tf.math.pow(estimation_imag, 2)))
    estimation_stft = tf.math.abs(estimation_stft)
    return tf.keras.losses.mean_squared_error(reference_stft, estimation_stft)
  return _loss


def phaseSensitiveSpectralApproximationLoss():
  """After backpropagation, estimation will be nan
  """
  def _loss(y_true, y_pred):
    reference_real = y_true[..., 0, :, :, :] # real/imag, ch, frame, freq
    reference_imag = y_true[..., 1, :, :, :]
    estimation_real = y_pred[..., 0, :, :, :]
    estimation_imag = y_pred[..., 1, :, :, :]

    reference_real = tf.cast(reference_real, dtype=tf.complex64)
    reference_imag = tf.cast(reference_imag, dtype=tf.complex64)
    estimation_real = tf.cast(estimation_real, dtype=tf.complex64)
    estimation_imag = tf.cast(estimation_imag, dtype=tf.complex64)

    reference_stft = reference_real + 1j*reference_imag
    estimation_stft = estimation_real + 1j*estimation_imag
    
    reference_stft_abs = tf.math.pow(tf.math.abs(reference_stft), 0.3)
    estimation_stft_abs = tf.math.pow(tf.math.abs(estimation_stft), 0.3)

    reference_stft_phase = tf.math.pow(reference_stft, 0.3)
    estimation_stft_phase = tf.math.pow(estimation_stft, 0.3)

    loss_phase = 0.113 * tf.math.pow(reference_stft_phase - estimation_stft_phase, 2)
    loss_phase = tf.cast(loss_phase, tf.float32)
    loss = tf.math.pow(reference_stft_abs - estimation_stft_abs, 2) + loss_phase
    return loss
  return _loss      
  

class SpeechMetric(tf.keras.metrics.Metric):
  """        
    [V] SI_SDR,     pass, after function check, value check
    [V] WB_PESQ,    pass, after function check, value check
    [ ] STOI,       fail, np.matmul, (15, 257) @ (257, 74) -> OMP: Error #131: Thread identifier invalid, zsh: abort
    [ ] NB_PESQ     fail, ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    [ ] SDR,        fail, MP: Error #131: Thread identifier invalid. zsh: abort      python train.py -> maybe batch related?

    [TODO] Verification, compared with pytorch
  """
  def __init__(self, metric, name='sisdr', **kwargs):
    super(SpeechMetric, self).__init__(name=name, **kwargs)
    self.metric = metric 
    self.score = self.add_weight(name=f"{name}_value", initializer='zeros')
    self.total = self.add_weight(name='total', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    # phase and amp
    # reference_amp = y_true[..., 0, :, :, :] # phase/amp, ch, frame, freq
    # reference_phase = y_true[..., 1, :, :, :]
    # estimation_amp = y_pred[..., 0, :, :, :]
    # estimation_phase = y_pred[..., 1, :, :, :]

    # reference_amp = tf.cast(reference_amp, dtype=tf.complex64)
    # reference_phase = tf.cast(reference_phase, dtype=tf.complex64)
    # estimation_amp = tf.cast(estimation_amp, dtype=tf.complex64)
    # estimation_phase = tf.cast(estimation_phase, dtype=tf.complex64)

    # reference_stft_librosa = reference_amp*tf.math.exp(-1j*reference_phase)
    # estimation_stft_librosa = estimation_amp*tf.math.exp(-1j*reference_phase)

    # real and imag
    # [TODO] Set const index of real/imag
    reference_real = y_true[..., 0, :, :, :] # real/imag, ch, frame, freq
    reference_imag = y_true[..., 1, :, :, :]
    estimation_real = y_pred[..., 0, :, :, :]
    estimation_imag = y_pred[..., 1, :, :, :]

    reference_real = tf.cast(reference_real, dtype=tf.complex64)
    reference_imag = tf.cast(reference_imag, dtype=tf.complex64)
    estimation_real = tf.cast(estimation_real, dtype=tf.complex64)
    estimation_imag = tf.cast(estimation_imag, dtype=tf.complex64)

    reference_stft_librosa = reference_real + 1j*reference_imag
    estimation_stft_librosa = estimation_real + 1j*estimation_imag

    window_fn = tf.signal.hamming_window

    reference = tf.signal.inverse_stft(
      reference_stft_librosa, frame_length=n_fft, frame_step=overlap,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step=overlap, forward_window_fn=window_fn))
    
    estimation = tf.signal.inverse_stft(
      estimation_stft_librosa, frame_length=n_fft, frame_step=overlap,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step=overlap, forward_window_fn=window_fn))

    self.score.assign_add(tf.py_function(func=self.metric, inp=[reference, estimation], Tout=tf.float32,  name='sisdr-metric')) # tf 2.x
    self.total.assign_add(1)
    
  def result(self):
    return self.score / self.total

def build_model_lstm(power=0.3):
  inputs = Input(shape=[2, 1, numSegments, numFeatures])  

  # x = tf.math.sqrt(tf.math.pow(inputs[..., 0, :, :, :], power)-tf.math.pow(inputs[..., 1, :, :, :], power)) # abs
  # x = inputs[..., 0, :, :, :]

  print("DEBUG", inputs[..., 0, :, :, :])
   
  inputs_real = tf.cast(inputs[..., 0, :, :, :], dtype=tf.complex64) # input이 아니라 layer(function) 을 쪼개는 것 처럼 보인다.
  inputs_imag = tf.cast(inputs[..., 1, :, :, :], dtype=tf.complex64) #

  inputs_clx = inputs_real + 1j*inputs_imag
  x = tf.math.abs(inputs_clx)
 
  mask = tf.squeeze(x, axis=1) # merge channel
  mask = MelSpec()(mask)
  mask = LSTM(256, activation='tanh', return_sequences=True)(mask)
  mask = LSTM(256, activation='tanh', return_sequences=True)(mask)
  
  mask = BatchNormalization()(mask)

  mask = Dense(128, activation='relu', use_bias=True, 
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(mask) # [TODO] check initialization method
  mask = Dense(128, activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(mask) # [TODO] check initialization method
  
  mask = InverseMelSpec()(mask)
  mask = tf.expand_dims(mask, axis=1) # expand channel
  mask = tf.expand_dims(mask, axis=1) # expand real and imag

  x = Multiply()([inputs, mask]) # X_bar = M (Hadamard product) |Y|exp(angle(Y)), Y is noisy
  model = Model(inputs=inputs, outputs=x)

  # optimizer = keras.optimizers.SGD(10e-4)
  optimizer = keras.optimizers.Adam(3e-4)
  #optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=3e-4)

  # model.compile(optimizer=optimizer, 
  #             loss= meanSquareError(), # 'mse'
  #             metrics=[keras.metrics.RootMeanSquaredError('rmse'), 
  #             ])

  model.compile(optimizer=optimizer, 
              loss= meanSquareError(),
              metrics=[
              SpeechMetric(metric=SI_SDR, name='sisdr'),
              # SpeechMetric(metric=WB_PESQ, name='wb_pesq'),
              ])
  return model


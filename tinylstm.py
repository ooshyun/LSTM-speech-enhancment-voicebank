import tensorflow as tf
import keras
import keras.layers
from librosa.filters import mel

def get_mel_filter(samplerate, n_fft, n_mels, fmin, fmax):
    mel_basis = mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return tf.convert_to_tensor(mel_basis)

class MaskTinyLSTM(keras.Model): 
    """
    If it define as function, then it shows the information
    """
    def __init__(self,
            original_dim,
            name="masktinylstm",
            **kwargs
        ):
        super(MaskTinyLSTM, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim

        self.lstm_layer_0 = keras.layers.LSTM(original_dim, activation='tanh', return_sequences=True)
        self.fully_connected_layer_0 = keras.layers.Dense(original_dim//2, activation='relu', use_bias=True,
                                                          kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.batch_norm_0 = keras.layers.BatchNormalization()

        self.lstm_layer_1 = keras.layers.LSTM(original_dim, activation='tanh')
        self.fully_connected_layer_1 = keras.layers.Dense(original_dim//2, activation='sigmoid', use_bias=True,
                                                          kernel_initializer='glorot_uniform', bias_initializer='zeros')

        # [TODO]
        # The network output, which shares the dimensionality of the input, is inverted using the corresponding transposed mel matrix to produce spectral mask M 

    def build(self, input_shape):
        # add initializer, Create the state of the layer (weights)
        # self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
        #                             initializer='uniform',
        #                             name='kernel')
        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units),
        #     initializer='uniform',
        #     name='recurrent_kernel')


        # For summary
        inputs = keras.layers.Input(shape=input_shape)
        self.outputs = self.call(tf.squeeze(inputs, axis=0))

        self.built = True
        super().build(input_shape)

    def call(self, input):
        lstm_0_out = self.lstm_layer_0(input)
        lstm_0_out = self.batch_norm_0(lstm_0_out) # [TODO] WHY use?
        lstm_0_out = self.fully_connected_layer_0(lstm_0_out)

        lstm_1_out = self.lstm_layer_1(lstm_0_out)
        lstm_1_out = self.fully_connected_layer_1(lstm_1_out)

        return lstm_1_out

class TinyLSTM(keras.Model):
    def __init__(self,
            original_dim,
            samplerate=16000,
            n_fft=512,
            n_mels=128,
            name="tinylstm",
            **kwargs
        ):
        super(TinyLSTM, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.mask = MaskTinyLSTM(original_dim=original_dim)        
        self.samplerate = samplerate
        # self.segment = # [TODO] Segmentation
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mel_matrix = None        
        if self.samplerate:
            self.mel_matrix = get_mel_filter(self.samplerate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=0, fmax=8000)
            # only tf.Variable has trainable attribute

    def build(self, input_shape):
        # add initializer        
        ...

        # For summary
        self.mask.build(input_shape=[input_shape[0], input_shape[2], self.n_mels])
        self.mask.summary()

        inputs = keras.layers.Input(shape=input_shape)
        self.outputs = self.call(tf.squeeze(inputs, axis=0)) # remove None of Input layer in the first of shape

        self.built = True
        super().build(input_shape)

    def call(self, inputs):        
        # When tf.saved_model.save(self.model, self.best_model), it occurs error as below,
        #     inputs = tf.reshape(inputs, [channel, batch_size, nframe, nsample])
        #     TypeError: Failed to convert elements of [1, None, 63, 257] to Tensor. Consider casting elements to a supported type. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.
        #     Call arguments received by layer "tinylstm" (type TinyLSTM):
        #       • inputs=tf.Tensor(shape=(None, 1, 63, 257), dtype=complex64)
        # remove channel shape because lstm input layer should be (batch, time, features)

        # Currently, it can run in mono
        # Remove channel layer, because rnn type input shape should be (batch, time, features),
        inputs = tf.squeeze(inputs, axis=1) 
        
        mode = inputs.dtype
        if mode == tf.complex64:
            input_amplitude = tf.abs(inputs)
        else:
            input_amplitude = inputs

        input_mel = tf.matmul(input_amplitude, tf.transpose(self.mel_matrix, perm=[1, 0]))

        mask = self.mask(input_mel)

        if self.mel_matrix != None:
            # In librosa, mel_basis = filters.mel(sr=sr, n_fft=n_fft, **kwargs)
            # return np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
            mask = tf.matmul(mask, self.mel_matrix)
        mask = tf.expand_dims(mask, axis=1)

        if mode == tf.complex64:
            input_real = tf.multiply(tf.math.real(inputs), mask)
            input_imag = tf.multiply(tf.math.imag(inputs), mask)
            estimate = tf.complex(input_real, input_imag)
        else:
            estimate = tf.multiply(inputs, mask)
        
        estimate = tf.expand_dims(estimate, axis=1)
        return estimate # [TODO] How to do the mask? refer ConvTasnet

    def test_summary(self, input_shape=(4, 64, 257)):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = keras.layers.Input(shape=input_shape)
        x = tf.squeeze(x, axis=0)
        model = keras.Model(inputs=x, outputs=self.call(x), name=self.name)
        print(model.summary())
        return model
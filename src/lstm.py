from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    LSTM,
    Dense,
    Layer,
    Multiply,
)

import tensorflow as tf
from keras import Model
import keras.regularizers
import keras.optimizers

import os
from .utils import load_json

from .metrics import (
    SI_SDR,
    # WB_PESQ,
    # SDR,
    # STOI,
    # NB_PESQ
)
from .loss import (
    convert_stft_from_amplitude_phase,
    mean_square_error_amplitdue_phase,
    mean_absolute_error_amplitdue_phase,
    ideal_amplitude_mask,
    phase_sensitive_spectral_approximation_loss,
    phase_sensitive_spectral_approximation_loss_bose,
)

class MelSpec(Layer):
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


class InverseMelSpec(Layer):
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


class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, metric, name="sisdr", **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.metric = metric
        self.metric_name = name
        self.score = self.add_weight(name=f"{name}_value", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        - Issue:
            loss function         -> (batch, ..., features) -> (batch, ..., 1) -> (batch, ...) -> losses_utils.ReductionV2.AUTO -> loss value
            current loss function -> (batch, 1, segments, features) -> (batch, 1, segments, 1) -> (batch, 1, segments) -> losses_utils.ReductionV2.AUTO -> loss value
            - [TODO] In metrics, we get the loss, which has shape (batch, 1, segments) -> How can losses_utils.ReductionV2.AUTO?
            - [SOL ] pass the train flag and reduce the loss by average
        """
        if self.metric == "mse":
            loss_function = mean_square_error_amplitdue_phase
        elif self.metric == "rmse":
            loss_function = mean_absolute_error_amplitdue_phase
        elif self.metric == "ideal-mag":
            loss_function = ideal_amplitude_mask
        elif self.metric == "psa":
            loss_function = phase_sensitive_spectral_approximation_loss
        elif self.metric == "psa_bose":
            loss_function = phase_sensitive_spectral_approximation_loss_bose
        else:
            raise NotImplementedError(f"Loss '{self.metric}' is not implemented")

        self.score.assign_add(
            tf.py_function(
                func=loss_function,
                inp=[y_true, y_pred, False],
                Tout=tf.float32,
                name=f"{self.metric_name}_metric",
            )
        )  # tf 2.x
        self.total.assign_add(1)

    def result(self):
        return self.score / self.total

    def get_config(self):
        config = super(CustomMetric, self).get_config()
        config.update(
            {
                "metric": self.metric,
                "metric_name": self.metric_name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SpeechMetric(tf.keras.metrics.Metric):
    """
    [V] SI_SDR,     pass, after function check, value check
    [V] WB_PESQ,    pass, after function check, value check
    [ ] STOI,       fail, np.matmul, (15, 257) @ (257, 74) -> OMP: Error #131: Thread identifier invalid, zsh: abort
    [ ] NB_PESQ     fail, ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    [ ] SDR,        fail, MP: Error #131: Thread identifier invalid. zsh: abort      python train.py -> maybe batch related?

    [TODO] Verification, compared with pytorch
    """

    def __init__(self, n_fft, hop_length, normalize, name="sisdr", **kwargs):
        super(SpeechMetric, self).__init__(name=name, **kwargs)
        self.metric_name = name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.score = self.add_weight(name=f"{name}_value", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.metric_name == "sisdr":
            func_metric = SI_SDR
        # elif self.metric_name == 'pesq':
        #   func_metric=PESQ
        # elif self.metric_name == 'stoi':
        #   func_metric=STOI
        else:
            raise NotImplementedError(
                f"Metric function '{self.metric}' is not implemented"
            )
        reference_stft_librosa = convert_stft_from_amplitude_phase(y_true)
        estimation_stft_librosa = convert_stft_from_amplitude_phase(y_pred)

        # related with preprocess normalized fft
        if self.normalize:
            reference_stft_librosa *= 2 * (
                reference_stft_librosa.shape[-1] - 1
            )  # [TODO] verfication
            estimation_stft_librosa *= 2 * (reference_stft_librosa.shape[-1] - 1)

        window_fn = tf.signal.hamming_window

        reference = tf.signal.inverse_stft(
            reference_stft_librosa,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            window_fn=tf.signal.inverse_stft_window_fn(
                frame_step=self.hop_length, forward_window_fn=window_fn
            ),
        )

        estimation = tf.signal.inverse_stft(
            estimation_stft_librosa,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            window_fn=tf.signal.inverse_stft_window_fn(
                frame_step=self.hop_length, forward_window_fn=window_fn
            ),
        )

        self.score.assign_add(
            tf.py_function(
                func=func_metric,
                inp=[reference, estimation],
                Tout=tf.float32,
                name=f"{self.metric_name}_metric",
            )
        )  # tf 2.x
        self.total.assign_add(1)

    def result(self):
        return self.score / self.total

    def get_config(self):
        config = super(SpeechMetric, self).get_config()
        config.update(
            {
                "metric_name": self.metric_name,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "normalize": self.normalize,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model_lstm(args, power=0.3):
    """
    # [TODO] How to print a tensor while training?
        Kernal Initialization
        Bias   Initizlization
    """
    inputs = Input(
        shape=[2, 1, args.model.n_segment, args.model.n_feature], name="input"
    )
    inputs_amp = inputs[..., 0, :, :, :]
    inputs_phase = inputs[..., 1, :, :, :]

    mask = tf.squeeze(inputs_amp, axis=1)  # merge channel
    
    mask = MelSpec(args)(mask)
    mask = LSTM(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)
    mask = LSTM(args.model.lstm_layer, activation="tanh", return_sequences=True)(mask)

    mask = BatchNormalization()(mask)

    mask = Dense(
        args.model.n_mels,
        activation="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform", #glorot_uniform
        bias_initializer="zeros",
    )(
        mask
    )  # [TODO] check initialization method
    mask = Dense(
        args.model.n_mels,
        activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(
        mask
    )  # [TODO] check initialization method

    mask = InverseMelSpec(args)(mask)

    mask = tf.expand_dims(mask, axis=1)  # expand channel

    outputs_amp = Multiply()(
        [inputs_amp, mask]
    )
    outputs = tf.stack(
        [outputs_amp, inputs_phase], axis=-4
    )  # ..., mag/phase, ch, num_frame, freq_bin

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
    elif args.optim.loss == "psa_bose":
        loss_function = phase_sensitive_spectral_approximation_loss_bose
    else:
        raise NotImplementedError(f"Optimizer {args.optim.optim} is not implemented")

    if args.model.path is not None:
        if "optimizer" in os.listdir(args.model.path):  # optimizer folder check
            tf.print("Optimizer Loading...")
            dummpy_model = build_model_lstm(args)
            optimizer_state = load_json(
                os.path.join(args.model.path, "optimizer/optim.json")
            )["optimizer"]
            dummy_batch_size = 1
            dummy_noise_tensor = tf.ones(
                shape=(
                    dummy_batch_size,
                    2,
                    1,
                    args.model.n_segment,
                    args.model.n_feature,
                )
            )
            dummy_clean_tensor = tf.ones(
                shape=(
                    dummy_batch_size,
                    2,
                    1,
                    args.model.n_segment,
                    args.model.n_feature,
                )
            )
            dummpy_model.compile(
                optimizer=optimizer,
                loss=loss_function,
            )
            dummpy_model.fit(
                x=dummy_noise_tensor, y=dummy_clean_tensor, batch_size=dummy_batch_size
            )

            del (
                dummpy_model,
                dummy_noise_tensor,
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
            n_fft=args.dset.n_fft,
            hop_length=args.dset.hop_length,
            normalize=args.dset.fft_normalize,
            name=metric_name,
        )
        for metric_name in args.model.metric
    ]
    metrics.append(CustomMetric(metric=args.optim.loss, name=args.optim.loss))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

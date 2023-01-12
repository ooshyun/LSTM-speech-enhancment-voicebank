import tensorflow as tf
import numpy as np
from pesq import pesq, cypesq
from pypesq import pesq as nb_pesq
from pystoi import stoi
from museval.metrics import bss_eval

from .loss import (
    mean_square_error_amplitdue_phase,
    mean_absolute_error_amplitdue_phase,
    ideal_amplitude_mask,
    phase_sensitive_spectral_approximation_loss,
    phase_sensitive_spectral_approximation_loss_bose,
)

def SDR(reference, estimation, sr=16000):
    """Signal to Distortion Ratio (SDR) from museval

    Reference
    ---------
    - https://github.com/sigsep/sigsep-mus-eval
    - Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
    IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462-1469.

    """
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 
    sdr_batch = np.empty(shape=(reference_numpy.shape[0], reference_numpy.shape[1]))

    for batch in range(reference_numpy.shape[0]):
        for ch in range(reference_numpy.shape[1]):
            sdr_batch[batch, ch], _, _, _, _ = bss_eval(
                reference_numpy[batch, ch], estimation_numpy[batch, ch]
            )
    sdr_batch = np.mean(sdr_batch)
    return sdr_batch


def SI_SDR(reference, estimation, sr=16000):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)。

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References:
        SDR- Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    reference_energy = np.sum(reference**2, axis=-1, keepdims=True)

    optimal_scaling = (
        np.sum(estimation*reference, axis=-1, keepdims=True) / (reference_energy + np.finfo(dtype=reference_energy.dtype).eps)
    )

    projection = optimal_scaling * reference
    noise = estimation - projection

    ratio = np.sum(projection**2, axis=-1) / (np.sum(noise**2, axis=-1) + np.finfo(dtype=reference_energy.dtype).eps)
    ratio = np.mean(ratio)
    return 10 * np.log10(ratio+np.finfo(dtype=reference_energy.dtype).eps)


def STOI(reference, estimation, sr=16000):
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 
    stoi_batch = np.empty(shape=(reference_numpy.shape[0], reference_numpy.shape[1]))
    for batch in range(reference_numpy.shape[0]):
        for ch in range(reference_numpy.shape[1]):
            stoi_batch[batch, ch] = stoi(
                reference_numpy[batch, ch],
                estimation_numpy[batch, ch],
                sr,
                extended=False,
            )

    stoi_batch = np.mean(stoi_batch)
    return stoi_batch


def WB_PESQ(reference, estimation, sr=16000):
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation        
    num_batch, num_channel = reference_numpy.shape[0], reference_numpy.shape[1]
    pesq_batch = np.empty(shape=(num_batch, num_channel))

    count_error = 0
    for batch in range(num_batch):
        for ch in range(num_channel):
            try:
                pesq_batch[batch, ch] = pesq(
                    sr,
                    reference_numpy[batch, ch],
                    estimation_numpy[batch, ch],
                    mode="wb",
                )
            except cypesq.NoUtterancesError:
                logging.info("cypesq.NoUtterancesError: b'No utterances detected'")
                count_error += 1
    if batch * num_channel - count_error > 0:
        pesq_batch = np.sum(pesq_batch) / (num_batch * num_channel - count_error)
    else:
        pesq_batch = 0
    return pesq_batch


def NB_PESQ(reference, estimation, sr=16000):
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 
    score = 0

    for ref_batch, est_batch in zip(reference_numpy, estimation_numpy):
        for ref_ch, est_ch in zip(ref_batch, est_batch):
            # print(ref_ch.shape, est_ch.shape, type(ref_ch), type(est_ch))
            # print(ref_ch, est_ch)
            score += nb_pesq(sr, ref_ch, est_ch)
    score /= (
        reference.shape[0] * reference.shape[1]
        if reference.shape[0] is not None
        else reference.shape[1]
    )
    score = np.squeeze(score)
    return score

class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, metric, name="mse", **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.metric = metric
        self.metric_name = name
        self.score = self.add_weight(name=f"{name}_value", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        """
        if self.metric == "mse":
            loss_function = mean_square_error_amplitdue_phase
        elif self.metric == "rmse":
            loss_function = mean_absolute_error_amplitdue_phase
        elif self.metric == "ideal-mag":
            loss_function = ideal_amplitude_mask
        elif self.metric == "psa":
            loss_function = phase_sensitive_spectral_approximation_loss
        elif self.metric == "psa-bose":
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
    [V] SI_SDR,     pass
    [V] WB_PESQ,    pass
    [ ] STOI,       fail, np.matmul, (15, 257) @ (257, 74) -> OMP: Error #131: Thread identifier invalid, zsh: abort
    [ ] NB_PESQ     fail, ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    [ ] SDR,        fail, MP: Error #131: Thread identifier invalid. zsh: abort      python train.py -> maybe batch related?

    [TODO] Verification, compared with pytorch
    """

    def __init__(self, model_name, n_fft, hop_length, normalize, name="sisdr", **kwargs):
        super(SpeechMetric, self).__init__(name=name, **kwargs)
        self.model_name = model_name
        self.metric_name = name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.score = self.add_weight(name=f"{name}_value", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.metric_name == "sisdr":
            func_metric=SI_SDR
        elif self.metric_name == 'wb-pesq':
            func_metric=WB_PESQ
        elif self.metric_name == 'stoi':
            func_metric=STOI
        elif self.metric_name == 'nb-pesq':
            func_metric=NB_PESQ
        elif self.metric_name == 'sdr':
            func_metric=SDR
        else:
            raise NotImplementedError(
                f"Metric function '{self.metric}' is not implemented"
            )

        if self.model_name not in ("unet", "conv-tasnet"):
            # related with preprocess normalized fft
            if self.normalize:
                y_true *= 2 * (y_true.shape[-1] - 1)  
                y_pred *= 2 * (y_pred.shape[-1] - 1)

            window_fn = tf.signal.hamming_window

            reference = tf.signal.inverse_stft(
                y_true,
                frame_length=self.n_fft,
                frame_step=self.hop_length,
                window_fn=tf.signal.inverse_stft_window_fn(
                    frame_step=self.hop_length, forward_window_fn=window_fn
                ),
            )

            estimation = tf.signal.inverse_stft(
                y_pred,
                frame_length=self.n_fft,
                frame_step=self.hop_length,
                window_fn=tf.signal.inverse_stft_window_fn(
                    frame_step=self.hop_length, forward_window_fn=window_fn
                ),
            )
        else:
            reference = y_true
            estimation = y_pred

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

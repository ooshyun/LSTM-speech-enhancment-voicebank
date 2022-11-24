import json
import yaml
import time
import numpy as np
import typing as tp
import librosa

# import sounddevice as sd
# import julius # for pytorch
from resampy import resample
import soundfile as sf
import tensorflow as tf

def limit_gpu_tf(memory_size):
    """Reference. https://www.tensorflow.org/guide/gpu
    """
    # device_name = tf.test.gpu_device_name()
    # if device_name != "/device:GPU:0":
    #     raise SystemError("GPU device not found")
    # print("Found GPU at: {}".format(device_name))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_size)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def inverse_stft_transform(stft_features, window_length, overlap):
    return librosa.istft(stft_features, win_length=window_length, hop_length=overlap)


def revert_features_to_audio(
    features, phase, window_length, overlap, cleanMean=None, cleanStd=None
):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return inverse_stft_transform(
        features, window_length=window_length, overlap=overlap
    )


def play(audio, sample_rate):
    # ipd.display(ipd.Audio(data=audio, rate=sample_rate))  # load a local WAV file
    # sd.play(audio, sample_rate, blocking=True)
    ...


def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind : ind + clean_audio.size]

    speech_power = np.sum(clean_audio**2)
    noise_power = np.sum(noiseSegment**2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio


def read_audio(filepath, sample_rate):
    # audio, sr = librosa.load(filepath, sr=sample_rate)
    # if normalize is True:
    #     div_fac = 1 / np.max(np.abs(audio)) / 3.0
    #     audio = audio * div_fac
    #     # audio = librosa.util.normalize(audio)
    audio, sr = sf.read(filepath)
    audio = resample(audio, sr, sample_rate)

    return audio, sr


def prepare_input_features(stft_features, numSegments, numFeatures):
    noisySTFT = np.concatenate(
        [stft_features[:, 0 : numSegments - 1], stft_features], axis=1
    )
    stftSegments = np.zeros(
        (numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1)
    )

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:, :, index] = noisySTFT[:, index : index + numSegments]
    return stftSegments


def get_input_features(predictorsList):
    predictors = []
    for noisy_stft_mag_features in predictorsList:
        # For CNN, the input feature consisted of 8 consecutive noisy
        # STFT magnitude vectors of size: 129 × 8,
        # TODO: duration: 100ms
        inputFeatures = prepare_input_features(noisy_stft_mag_features)
        # print("inputFeatures.shape", inputFeatures.shape)
        predictors.append(inputFeatures)

    return predictors


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tf_feature(noise_stft_mag_features, clean_stft_magnitude, noise_stft_phase):
    noise_stft_mag_features = noise_stft_mag_features.astype(np.float32).tostring()
    clean_stft_magnitude = clean_stft_magnitude.astype(np.float32).tostring()
    noise_stft_phase = noise_stft_phase.astype(np.float32).tostring()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "noise_stft_phase": _bytes_feature(noise_stft_phase),
                "noise_stft_mag_features": _bytes_feature(noise_stft_mag_features),
                "clean_stft_magnitude": _bytes_feature(clean_stft_magnitude),
            }
        )
    )
    return example


def get_tf_feature_mag_phase_pair(
    noisy_stft_magnitude, clean_stft_magnitude, noise_stft_phase, clean_stft_phase
):
    noisy_stft_magnitude = noisy_stft_magnitude.astype(np.float32).tostring()
    clean_stft_magnitude = clean_stft_magnitude.astype(np.float32).tostring()
    noise_stft_phase = noise_stft_phase.astype(np.float32).tostring()
    clean_stft_phase = clean_stft_phase.astype(np.float32).tostring()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "noisy_stft_magnitude": _bytes_feature(noisy_stft_magnitude),
                "clean_stft_magnitude": _bytes_feature(clean_stft_magnitude),
                "noise_stft_phase": _bytes_feature(noise_stft_phase),
                "clean_stft_phase": _bytes_feature(clean_stft_phase),
            }
        )
    )
    # print(_bytes_feature(b'test_string'))
    # print(_bytes_feature(u'test_bytes'.encode('utf-8')))
    return example


def get_tf_feature_sample_pair(noisy, clean):
    noisy = noisy.astype(np.float32).tostring()
    clean = clean.astype(np.float32).tostring()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={"noisy": _bytes_feature(noisy), "clean": _bytes_feature(clean)}
        )
    )
    return example


def stft_tensorflow(wav, nfft, hop_length, center=True, normalize=True):
    if center:
        padding = [(0, 0) for _ in range(len(wav.get_shape()))]
        padding[-1] = (int(nfft // 2), int(nfft // 2))
        wav = tf.pad(wav, padding, mode="constant")

    window_fn = tf.signal.hann_window
    wav_stft = tf.signal.stft(
        wav,
        frame_length=nfft,
        frame_step=hop_length,
        window_fn=window_fn,
        pad_end=False,
    )
    if normalize:
        wav_stft = tf.divide(wav_stft, nfft) # normalize

    # if using inverse stft,
    # inverse_stft = tf.signal.inverse_stft(
    #   wav_stft, frame_length=n_fft, frame_step=overlap,
    #   window_fn=tf.signal.inverse_stft_window_fn(
    #      frame_step=overlap, forward_window_fn=window_fn))

    # [TODO] Phase aware process
    wav_stft_mag_features = tf.abs(wav_stft)
    wav_stft_phase = tf.experimental.numpy.angle(wav_stft)
    wav_stft_real = tf.math.real(wav_stft)
    wav_stft_imag = tf.math.imag(wav_stft)

    return wav_stft_mag_features, wav_stft_phase, wav_stft_real, wav_stft_imag


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(TimeHistory, self).__init__()
        self.filepath = filepath

    def on_train_begin(self, logs={}):
        self.epoch_times = []
        self.batch_times = []
        self.train_time = []
        self.train_time.append(time.perf_counter())

    def on_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.perf_counter()

    def on_batch_end(self, batch, logs={}):
        self.batch_times.append(time.perf_counter() - self.batch_time_start)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time.perf_counter() - self.epoch_time_start)

    def on_train_end(self, logs={}):
        self.train_time.append(time.perf_counter())

        log_file_path = self.filepath
        with open(log_file_path, "w") as tmp:
            tmp.write(f"  Total time\n")
            tmp.write(f"start    : {self.train_time[0]} sec\n")
            tmp.write(f"end      : {self.train_time[1]} sec\n")
            tmp.write(f"duration : {self.train_time[1]- self.train_time[0]} sec\n")

            tmp.write(f"  Epoch time, {len(self.epoch_times)}\n")
            for epoch, t in enumerate(self.epoch_times):
                tmp.write(f"{epoch} : {t}\n")

            tmp.write(f"  Batch time, {len(self.batch_times)}\n")

            for num, t in enumerate(self.batch_times):
                tmp.write(f"{t} ")
                if num % 100 == 99:
                    tmp.write(f"\n")


def load_yaml(path: str, *args, **kwargs) -> dict:
    with open(path, "r") as tmp:
        try:
            return dict2obj(yaml.safe_load(tmp))

        except yaml.YAMLError as exc:
            print(exc)


# declaring a class
class Config:
    pass


def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
        return d

    # constructor of the class passed to obj
    obj = Config()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
    return obj


def load_json(path: str, *args, **kwargs) -> tp.Dict[str, list]:
    """Tested at study/test_save_optimizer.py"""
    with open(path, "r") as tmp:
        data: dict = json.load(tmp, *args, **kwargs)
        for key, value in data.items():
            if key == "args":
                continue
            else:
                for ival, val in enumerate(value):
                    data[key][ival] = np.array(
                        val,
                        dtype=type(val) if not isinstance(val, list) else type(val[0]),
                    )
    return data


def save_json(data: tp.Dict[str, tp.List[np.ndarray]], path: str, *args, **kwargs):
    """Tested at study/test_save_optimizer.py"""
    for key, value in data.items():
        if isinstance(value, Config):
            data[key] = obj2dict(value)

    with open(path, "w") as tmp:
        json.dump(data, tmp, cls=NumpyEncoder, *args, **kwargs)


def obj2dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        element = []
        if isinstance(val, list):
            for item in val:
                element.append(obj2dict(item))
        else:
            element = obj2dict(val)
        result[key] = element
    return result


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    Tested at study/test_save_optimizer.py
    Reference. https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def prevent_clip(wav, mode="rescale"):
    """
    different strategies for avoiding raw clipping.
    """
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == "rescale":
        wav = wav / max(1.01 * np.max(np.abs(wav)), 1)
    elif mode == "clamp":
        wav = np.clip(wav, a_min=-0.99, a_max=0.99)
    elif mode == "tanh":
        wav = np.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def encode_normalize(wav, normalize, metadata=None):
    """
    Refernence. https://developers.google.com/machine-learning/data-prep/transform/normalization
    """
    eps = 1e-12
    if metadata is None:
        if normalize == "z-score":
            mean = np.expand_dims(np.mean(wav, axis=-1), axis=-1)
            std = np.expand_dims(np.std(wav, axis=-1), axis=-1)
            wav = (wav - mean) / (std + eps)
        elif normalize == "min-max":  # linear-scale'
            min = np.expand_dims(np.min(wav, axis=-1), axis=-1)
            max = np.expand_dims(np.max(wav, axis=-1), axis=-1)
            wav = (wav - min) / (max - min + eps)
        elif normalize == "log-scale":
            wav = np.log(np.abs(wav) + eps)
        elif normalize == "clip":
            NotImplementedError
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {normalize}")
        return wav
    else:
        return encode_normalize_with_metadata(wav, normalize, metadata)


def encode_normalize_with_metadata(wav, normalize, metadata):
    if normalize == "z-score":
        wav = (wav - metadata["mean"]) / (metadata["std"])
    elif normalize == "linear-scale":
        wav = (wav - metadata["min"]) / (metadata["max"] - metadata["min"])
    elif normalize == "clip":
        NotImplementedError
    elif normalize == "log-scale":
        NotImplementedError
    else:
        raise ValueError(f"Invalid normalize method: {normalize}")
    return wav


def decode_normalize(wav, normalize, metadata):
    """
    Refernence. https://developers.google.com/machine-learning/data-prep/transform/normalization
    """
    if normalize == "z-score":
        wav = wav * metadata["std"] + metadata["mean"]
    elif normalize == "linear-scale":
        wav = wav * (metadata["max"] - metadata["min"]) + metadata["min"]
    elif normalize == "clip":
        NotImplementedError
    elif normalize == "log-scale":
        NotImplementedError
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Invalid normalize method: {normalize}")
    return wav


def segment_audio(audio, sample_rate, segment):
    # sec
    length_audio = audio.shape[-1]
    num_sample_segment = int(segment * sample_rate)
    nsegment = length_audio // num_sample_segment

    if length_audio / sample_rate < segment:
        audio = np.pad(
            audio, ((0, 0), (0, num_sample_segment - length_audio)), "constant"
        )
    elif length_audio % num_sample_segment != 0:
        audio = audio[: nsegment * num_sample_segment]

    newshape = list(audio.shape)
    if len(newshape) == 1:
        newshape = [nsegment, num_sample_segment]
    else:
        newshape = [audio.shape[:-1]] + [nsegment, num_sample_segment]
    audio = np.reshape(audio, newshape=newshape)  # ..., segment, samples

    audio_segments = None
    for isegment in range(audio.shape[-2]):
        if audio_segments is None:
            audio_segments = audio[..., isegment, :]
        else:
            if audio_segments.shape != audio[..., isegment, :].shape:
                buf_audio_segment = np.expand_dims(audio[..., isegment, :], axis=0)
                audio_segments = np.concatenate(
                    [audio_segments, buf_audio_segment], axis=0
                )
            else:
                audio_segments = np.stack(
                    [audio_segments, audio[..., isegment, :]], axis=0
                )
    return audio

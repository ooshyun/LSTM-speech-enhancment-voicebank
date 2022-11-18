import tensorflow as tf
from librosa.filters import mel

def get_mel_filter(samplerate, n_fft, n_mels, fmin, fmax):
    mel_basis = mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return tf.convert_to_tensor(mel_basis)

mel_matrix = get_mel_filter(samplerate=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000)

print(mel_matrix.shape)
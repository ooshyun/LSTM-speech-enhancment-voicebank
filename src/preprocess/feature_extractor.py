import librosa
import scipy.signal as signal


class FeatureExtractor:
    def __init__(self, audio, *, windowLength, hop_length, sample_rate):
        self.audio = audio
        self.fft_length = windowLength
        self.window_length = windowLength
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window = signal.hanning(self.window_length, sym=False) # sym true: filter, false: spectral analysis

    def get_stft_spectrogram(self, center):
        return librosa.stft(self.audio, n_fft=self.fft_length, win_length=self.window_length, hop_length=self.hop_length,
                            window=self.window, center=center)

    def get_audio_from_stft_spectrogram(self, stft_features, center):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.hop_length,
                             window=self.window, center=center)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect',
                                              n_fft=self.fft_length, hop_length=self.hop_length, center=True)

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.fft_length,
                                                    hop_length=self.hop_length,
                                                    win_length=self.window_length, window=self.window,
                                                    center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None)

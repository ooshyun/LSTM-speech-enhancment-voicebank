import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
from src.preprocess.VoiceBankDEMAND import VoiceBandDEMAND
from src.preprocess.feature_extractor import FeatureExtractor
from src.utils import read_audio, load_yaml
from src.distrib import load_dataset, load_model

# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from tensorflow.python.client import device_lib

def inference(clean_file, noisy_file, args):
    # 1. Set Paramter
    device_lib.list_local_devices()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.name
    save_path = args.dset.save_path
    flag_fft = args.dset.fft
    nfft = args.dset.n_fft
    hop_length = args.dset.hop_length
    center = args.dset.center
    num_features = args.model.n_feature
    num_segments = args.model.n_segment
    normalization = args.dset.normalize
    fft_normalization = args.dset.fft_normalize
    top_db = args.dset.top_db
    train_split = int(args.dset.split*100)
    sample_rate = args.dset.sample_rate
    win_length = args.dset.win_length

    # 3. Build and Load Model
    model = load_model(args)

    # 4. Load audio files
    clean_audio, sr = read_audio(clean_file, sample_rate)
    noisy_audio, sr = read_audio(noisy_file, sample_rate)

    mean = np.mean(noisy_audio)
    std = np.std(noisy_audio)

    noisy_audio_norm = (noisy_audio - mean) / std

    clean_audio_feature_extractor = FeatureExtractor(clean_audio, windowLength=win_length, hop_length=hop_length, sample_rate=sr)
    stft_features = clean_audio_feature_extractor.get_stft_spectrogram(center)
    stft_features = np.abs(stft_features)

    print("Min:", np.min(stft_features),"Max:",np.max(stft_features))

    noise_audio_feature_extractor = FeatureExtractor(noisy_audio_norm, windowLength=win_length, hop_length=hop_length, sample_rate=sr)
    noise_stft_features = noise_audio_feature_extractor.get_stft_spectrogram(center)
    noise_stft_features /= nfft

    noisy_phase = np.angle(noise_stft_features)
    noise_amplitude = np.abs(noise_stft_features)


    def _prepare_input_features(stft_features, numSegments, numFeatures):
        noisySTFT = np.concatenate([stft_features[:, 0:numSegments - 1], stft_features], axis=1)
        stftSegments = np.zeros((numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))

        for index in range(noisySTFT.shape[1] - numSegments + 1):
            stftSegments[:, :, index] = noisySTFT[:, index:index + numSegments]
        return stftSegments
    

    def revert_features_to_audio(features, phase, mean=None, std=None):
        features = np.squeeze(features)

        # features = librosa.db_to_power(features)
        features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

        features = np.transpose(features, (1, 0))
        features *= nfft
        estimated_audio = noise_audio_feature_extractor.get_audio_from_stft_spectrogram(features, center)
        
        # scale the outpus back to the original range
        if mean and std:
            estimated_audio = std * estimated_audio + mean
        
        return estimated_audio


    predictors_amp = _prepare_input_features(noise_amplitude, num_segments, num_features)
    predictors_amp = np.reshape(predictors_amp, (predictors_amp.shape[0], predictors_amp.shape[1], predictors_amp.shape[2]))
    predictors_amp = np.transpose(predictors_amp, (2, 0, 1)).astype(np.float32)
    predictors_amp = np.transpose(predictors_amp, (0, 2, 1))

    predictors_phase = _prepare_input_features(noisy_phase, num_segments, num_features)
    predictors_phase = np.reshape(predictors_phase, (predictors_phase.shape[0], predictors_phase.shape[1], predictors_phase.shape[2]))
    predictors_phase = np.transpose(predictors_phase, (2, 0, 1)).astype(np.float32)
    predictors_phase = np.transpose(predictors_phase, (0, 2, 1))

    predictors = np.stack([predictors_amp, predictors_phase], axis=1)
    predictors = np.expand_dims(predictors, axis=2)

    estimation_stft = model.predict(predictors)
    estimation_stft = estimation_stft[..., -1, :]
    estimation_stft = np.squeeze(estimation_stft)

    if model_name == 'lstm':
        estimation = revert_features_to_audio(estimation_stft[:, 0, :], estimation_stft[:, 1, :], mean, std)

    print("Min:", np.min(estimation),"Max:",np.max(estimation))

    return clean_audio, noisy_audio, estimation


if __name__ == "__main__":
    # SHOULD PUT model path
    model_path = Path(f'./history/221122-1127/data/20221123-183326')
    path_conf = os.path.join(model_path, "config.yaml")
    args = load_yaml(path_conf)
    
    args.model.path = model_path.as_posix()
    args.dset.wav = "/Users/seunghyunoh/workplace/research/NoiseReduction/Tiny-SpeechEnhancement/data/VoiceBankDEMAND/DS_10283_2791"

    test_dataset_voicebank = VoiceBandDEMAND(args.dset.wav, val_dataset_percent=0.3)
    clean_test_filenames, noisy_test_filenames = test_dataset_voicebank.get_test_filenames()
    clean_file = clean_test_filenames[0]
    noisy_file = noisy_test_filenames[0]

    clean_audio, noisy_audio, estimation = inference(clean_file, noisy_file, args)

    # plot
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

    # ax1.plot(clean_audio)
    # ax1.set_title("Clean Audio")

    # ax2.plot(noisy_audio)
    # ax2.set_title("Noisy Audio")

    # ax3.plot(estimation)
    # ax3.set_title("Denoised Audio")

    # plt.grid()
    # plt.show()

    import os
    import soundfile as sf
    filename = clean_file.split('/')[-1].split('.')[0]
    save_path = model_path / "audio" / filename
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)
    
    save_path = save_path.as_posix()

    estimate_file_name = f"{clean_file.split('/')[-1].split('.')[0]}_estimate.wav"
    clean_file_name = f"{clean_file.split('/')[-1].split('.')[0]}_clean.wav"
    noisy_file_name = f"{clean_file.split('/')[-1].split('.')[0]}_noisy.wav"

    clean_file = os.path.join(save_path, clean_file_name)
    noisy_file = os.path.join(save_path, noisy_file_name)
    estimation_file = os.path.join(save_path, estimate_file_name)

    sf.write(clean_file, clean_audio, args.dset.sample_rate)
    sf.write(noisy_file, noisy_audio, args.dset.sample_rate)
    sf.write(estimation_file, estimation, args.dset.sample_rate)


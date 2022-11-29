import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import os
import json
import soundfile as sf
from pathlib import Path
import tensorflow as tf
import numpy as np
from src.preprocess.VoiceBankDEMAND import VoiceBandDEMAND
from src.preprocess.feature_extractor import FeatureExtractor
from src.utils import read_audio, load_yaml
from src.distrib import load_dataset, load_model

# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from tensorflow.python.client import device_lib

def inference(clean_file, noisy_file, args, return_metric=False):
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

    mean_noisy = np.mean(noisy_audio)
    std_noisy = np.std(noisy_audio)
    noisy_audio_norm = (noisy_audio - mean_noisy) / std_noisy

    noisy_audio_feature_extractor = FeatureExtractor(noisy_audio_norm, windowLength=win_length, hop_length=hop_length, sample_rate=sr)
    noisy_stft_features = noisy_audio_feature_extractor.get_stft_spectrogram(center)

    noisy_stft_features /= nfft

    # Paper: Besides, spectral phase was not used in the training phase.
    # At reconstruction, noisy spectral phase was used instead to
    # perform in- verse STFT and recover human speech.

    noisy_phase = np.angle(noisy_stft_features)
    noisy_amplitude = np.abs(noisy_stft_features)

    def _prepare_input_features(stft_features, numSegments, numFeatures):
        stftSegments = np.zeros((numFeatures, numSegments, stft_features.shape[1]))
        for index in range(stft_features.shape[1]-numSegments+1):
            stftSegments[..., index] = stft_features[..., index:index + numSegments]
        return stftSegments

    def _prepare_input_features_zero_filled(stft_features, numSegments, numFeatures):
        noisySTFT = np.concatenate([np.zeros_like(stft_features[:, 0:numSegments - 1]), stft_features], axis=1)
        stftSegments = np.zeros((numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))
        for index in range(noisySTFT.shape[1] - numSegments + 1):
            stftSegments[..., index] = noisySTFT[..., index:index + numSegments]
        return stftSegments


    def revert_features_to_audio(features, phase, mean=None, std=None):
        features = np.squeeze(features)

        # features = librosa.db_to_power(features)
        features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

        features = np.transpose(features, (1, 0))
        features *= nfft
        estimated_audio = noisy_audio_feature_extractor.get_audio_from_stft_spectrogram(features, center)
        
        # scale the outpus back to the original range
        if mean and std:
            estimated_audio = std * estimated_audio + mean
        
        return estimated_audio

    noisy_ampltidue_input = _prepare_input_features_zero_filled(noisy_amplitude, num_segments, num_features)
    noisy_ampltidue_input = np.transpose(noisy_ampltidue_input, (2, 1, 0)).astype(np.float32)

    noisy_phase_input = _prepare_input_features_zero_filled(noisy_phase, num_segments, num_features)
    noisy_phase_input = np.transpose(noisy_phase_input, (2, 1, 0)).astype(np.float32)

    noisy_input = np.stack([noisy_ampltidue_input, noisy_phase_input], axis=1)
    noisy_input = np.expand_dims(noisy_input, axis=2)
    estimation_amp_phase = model.predict(noisy_input)

    estimation_amp_phase = estimation_amp_phase[..., -1, :]
    estimation_amp_phase = np.squeeze(estimation_amp_phase)

    if model_name == 'lstm':
        estimation = revert_features_to_audio(estimation_amp_phase[:, 0, :], estimation_amp_phase[:, 1, :], mean_noisy, std_noisy)

    print("Min:", np.min(estimation),"Max:",np.max(estimation))

    metric_sisdr = None
    if return_metric:
        filename = clean_file.split('/')[-1].split('.')[0]
        metric_sisdr = {filename:{}}
        # For metrics
        mean_clean = np.mean(clean_audio)
        std_clean = np.std(clean_audio)
        clean_audio_norm = (clean_audio - mean_clean) / std_clean
        clean_audio_feature_extractor = FeatureExtractor(clean_audio_norm, windowLength=win_length, hop_length=hop_length, sample_rate=sr)
        clean_stft_features = clean_audio_feature_extractor.get_stft_spectrogram(center)
        clean_stft_features /= nfft
        clean_phase = np.angle(clean_stft_features)
        clean_amplitude = np.abs(clean_stft_features)

        noisy_bypass = noisy_input[..., -1, :]
        noisy_bypass = np.squeeze(noisy_bypass)
        noisy_bypass = revert_features_to_audio(noisy_bypass[:, 0, :], noisy_bypass[:, 1, :], mean_noisy, std_noisy)

        clean_ampltidue_input = _prepare_input_features_zero_filled(clean_amplitude, num_segments, num_features)
        clean_ampltidue_input = np.transpose(clean_ampltidue_input, (2, 1, 0)).astype(np.float32)

        clean_phase_input = _prepare_input_features_zero_filled(clean_phase, num_segments, num_features)
        clean_phase_input = np.transpose(clean_phase_input, (2, 1, 0)).astype(np.float32)

        clean_input = np.stack([clean_ampltidue_input, clean_phase_input], axis=1)
        clean_input = np.expand_dims(clean_input, axis=2)

        clean_bypass = clean_input[..., -1, :]
        clean_bypass = np.squeeze(clean_bypass)
        clean_bypass = revert_features_to_audio(clean_bypass[:, 0, :], clean_bypass[:, 1, :], mean_clean, std_clean)

        from src.metrics import SI_SDR

        metric = SI_SDR

        estimation_metric = estimation

        new_shape = list(clean_bypass.shape)
        nsegment_metric = int(new_shape[-1]//sample_rate)
        new_shape += [0]
        new_shape[-2:] = nsegment_metric, sample_rate

        clean_bypass = clean_bypass[:nsegment_metric*sample_rate]
        clean_bypass = np.reshape(clean_bypass, new_shape)
        noisy_bypass = noisy_bypass[:nsegment_metric*sample_rate]
        noisy_bypass = np.reshape(noisy_bypass, new_shape)
        estimation_metric = estimation[:nsegment_metric*sample_rate]
        estimation_metric = np.reshape(estimation_metric, new_shape)

        metric_sisdr[filename]['sisdr'] = {}
        for nseg, (clean_seg, noisy_seg, est_seg) in enumerate(zip(clean_bypass, noisy_bypass, estimation_metric)):
            sisdr_prev = metric(clean_seg, noisy_seg, sample_rate)
            sisdr_after = metric(clean_seg, est_seg, sample_rate)
            metric_sisdr[filename]['sisdr'][(nseg)] = [sisdr_prev, sisdr_after]
            
        sisdr_prev = metric(clean_bypass, noisy_bypass, sample_rate)
        sisdr_after = metric(clean_bypass, estimation_metric, sample_rate)
        metric_sisdr[filename]['sisdr']['total'] = [sisdr_prev, sisdr_after]

    return clean_audio, noisy_audio, estimation, metric_sisdr


if __name__ == "__main__":
    # SHOULD PUT model path
    model_path = Path(f'./history/221122-1127/data/20221123-183326')
    path_conf = os.path.join(model_path, "config.yaml")
    args = load_yaml(path_conf)
    
    args.model.path = model_path.as_posix()
    args.dset.wav = "/Users/seunghyunoh/workplace/research/NoiseReduction/Tiny-SpeechEnhancement/data/VoiceBankDEMAND/DS_10283_2791"

    test_dataset_voicebank = VoiceBandDEMAND(args.dset.wav, val_dataset_percent=0.3)
    clean_test_filenames, noisy_test_filenames = test_dataset_voicebank.get_test_filenames()
    
    writen = True

    for clean_file, noisy_file in zip(clean_test_filenames, noisy_test_filenames):
        # clean_file = clean_test_filenames[0]
        # noisy_file = noisy_test_filenames[0]

        clean_audio, noisy_audio, estimation, metrics_file = inference(clean_file, noisy_file, args, return_metric=True)


        if writen:
            filename = clean_file.split('/')[-1].split('.')[0]
            save_path = model_path / "audio"
            file_save_path = save_path / filename
            metric_save_path = file_save_path / "result_metric.json"

            if not file_save_path.is_dir():
                file_save_path.mkdir(parents=True, exist_ok=True)

            estimate_file_name = f"{clean_file.split('/')[-1].split('.')[0]}_estimate.wav"
            clean_file_name = f"{clean_file.split('/')[-1].split('.')[0]}_clean.wav"
            noisy_file_name = f"{clean_file.split('/')[-1].split('.')[0]}_noisy.wav"
            
            clean_file = os.path.join(file_save_path, clean_file_name)
            noisy_file = os.path.join(file_save_path, noisy_file_name)
            estimation_file = os.path.join(file_save_path, estimate_file_name)

            sf.write(clean_file, clean_audio, args.dset.sample_rate)
            sf.write(noisy_file, noisy_audio, args.dset.sample_rate)
            sf.write(estimation_file, estimation, args.dset.sample_rate)

            if metrics_file is not None:
                with open(metric_save_path, 'w') as tmp:
                    print(metrics_file)
                    json.dump(metrics_file, tmp,  indent=4)
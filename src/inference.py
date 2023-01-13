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
from src.utils import read_audio, load_yaml, limit_gpu_tf
from src.distrib import load_model

# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from tensorflow.python.client import device_lib

def _prepare_input_features(feature, numSegments, numFeatures):
    segments = np.zeros((numFeatures, numSegments, feature.shape[1]))
    for index in range(feature.shape[1]-numSegments+1):
        segments[..., index] = feature[..., index:index + numSegments]
    return segments

def _prepare_input_stft_zero_filled(stft, numSegments, numFeatures, pad=True):
    if pad:
        stft_padded = np.concatenate([np.zeros_like(stft[:, 0:numSegments - 1]), stft], axis=-1)

    segments = np.zeros((numFeatures, numSegments, stft_padded.shape[1] - numSegments + 1), dtype=stft.dtype)
    
    for index in range(stft_padded.shape[1] - numSegments + 1):
        segments[..., index] = stft_padded[..., index:index + numSegments]
    
    return segments

def _prepare_input_wav_zero_filled(wav, num_feature, stride):
    assert wav.shape[-1] >= num_feature, "the length of data is too short comparing the number of features..."

    if (wav.shape[-1] - num_feature) % stride != 0:
        npad = stride*((wav.shape[-1] - num_feature)//stride + 1) - (wav.shape[-1] - num_feature)
        padding = np.zeros(shape=(len(wav.shape)*2), dtype=int)
        padding[-1] = npad
        padding = np.reshape(padding, newshape=(len(wav.shape), 2))
        wav_padded = np.pad(wav, pad_width=padding, mode="constant", constant_values=0)
    else:
        wav_padded = wav

    num_segment = (wav_padded.shape[-1] - num_feature) // stride +1

    shape = list(wav.shape)
    shape = [num_segment] + shape[:-1] + [num_feature]
    wav_segments = np.zeros(shape=shape, dtype=wav.dtype)
    
    for index in range(num_segment):
        wav_segments[index, ...] = wav_padded[..., index*stride:index*stride+num_feature]
    
    return wav_segments


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
    normalization = args.dset.normalize
    top_db = args.dset.top_db
    train_split = int(args.dset.split*100)
    sample_rate = args.dset.sample_rate
    win_length = args.dset.win_length
    segment = args.dset.segment
    num_segments = int(segment*sample_rate//hop_length + 1)

    # 3. Build and Load Model
    model = load_model(args)

    # 4. Load audio files
    clean_audio, sr = read_audio(clean_file, sample_rate)
    noisy_audio, sr = read_audio(noisy_file, sample_rate)

    mean_noisy = np.mean(noisy_audio)
    std_noisy = np.std(noisy_audio)
    noisy_audio_norm = (noisy_audio - mean_noisy) / std_noisy

    if model_name in ("unet", "conv-tasnet"):
        num_feature = int(sample_rate*segment)
        stride = 256
        noisy_input = _prepare_input_wav_zero_filled(noisy_audio_norm, num_feature=num_feature, stride=stride)
        noisy_input = np.expand_dims(noisy_input, axis=1)
    else:
        noisy_audio_feature_extractor = FeatureExtractor(noisy_audio_norm, windowLength=win_length, hop_length=hop_length, sample_rate=sr)
        noisy_stft_features = noisy_audio_feature_extractor.get_stft_spectrogram(center)

        noisy_stft_features /= nfft

        noisy_input = _prepare_input_stft_zero_filled(noisy_stft_features, num_segments, num_features)
        noisy_input = np.transpose(noisy_input, axes=(2, 1, 0))
        noisy_input = np.expand_dims(noisy_input, axis=1)
        
    output = model.predict(noisy_input)
    
    if model_name in ("unet", "conv-tasnet"):
        shape = list(noisy_audio_norm.shape)
        shape = shape[:-1] + [num_feature + stride*(output.shape[0]-1)]
        estimation = np.zeros(shape=shape, dtype=noisy_audio_norm.dtype)
        estimation[..., :num_feature] = output[0, ...]
        for ibatch in range(output.shape[0]-1):
            curr_loc = num_feature + stride*ibatch
            estimation[..., curr_loc: curr_loc+stride] = output[ibatch+1, ..., -stride:]

        estimation = estimation[..., :noisy_audio_norm.shape[-1]]
        estimation = estimation*mean_noisy + std_noisy
    else:
        output = output[..., -1, :]
        output = np.squeeze(output)

        def revert_features_to_audio(stft, mean=None, std=None):
            stft = np.transpose(stft, (1, 0))
            stft *= nfft
            estimated_audio = noisy_audio_feature_extractor.get_audio_from_stft_spectrogram(stft, center)
            
            # scale the outpus back to the original range
            if mean and std:
                estimated_audio = std * estimated_audio + mean
            
            return estimated_audio
        estimation = revert_features_to_audio(output, mean_noisy, std_noisy)
    
    print("Min:", np.min(estimation),"Max:",np.max(estimation))

    metric_sisdr = None
    if return_metric:
        filename = clean_file.split('/')[-1].split('.')[0]
        metric_sisdr = {filename:{}}
        
        if model_name in ("unet", "conv-tasnet"):
            noisy_bypass = mean_noisy*noisy_audio_norm + std_noisy
            clean_bypass = clean_audio
        else:
            mean_clean = np.mean(clean_audio)
            std_clean = np.std(clean_audio)
            clean_audio_norm = (clean_audio - mean_clean) / std_clean
            clean_audio_feature_extractor = FeatureExtractor(clean_audio_norm, windowLength=win_length, hop_length=hop_length, sample_rate=sr)
            clean_stft_features = clean_audio_feature_extractor.get_stft_spectrogram(center)
            clean_stft_features /= nfft
            
            noisy_bypass = noisy_input[..., -1, :]
            noisy_bypass = np.squeeze(noisy_bypass)
            noisy_bypass = revert_features_to_audio(noisy_bypass, mean_noisy, std_noisy)

            clean_input = _prepare_input_stft_zero_filled(clean_stft_features, num_segments, num_features)
            clean_input = np.transpose(clean_input, (2, 1, 0))
            clean_input = np.expand_dims(clean_input, axis=1)

            clean_bypass = clean_input[..., -1, :]
            clean_bypass = np.squeeze(clean_bypass)
            clean_bypass = revert_features_to_audio(clean_bypass, mean_clean, std_clean)

        estimation_metric = estimation

        from src.model.metrics import SI_SDR

        metric = SI_SDR

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

def main(gpu_size, path_conf):
    limit_gpu_tf(gpu_size) # 12G
    args = load_yaml(path_conf)
    model_path = args.model.path

    path_conf = os.path.join(model_path, "config.yaml")
    args = load_yaml(path_conf)
    
    test_dataset = VoiceBandDEMAND(args.test.wav, val_dataset_percent=0)
    clean_test_filenames, noisy_test_filenames = test_dataset.get_test_filenames()

    for step, (clean_file, noisy_file) in enumerate(zip(clean_test_filenames, noisy_test_filenames)):
        clean_audio, noisy_audio, estimation, metrics_file = inference(clean_file, noisy_file, args, return_metric=True)

        if args.test.save:
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
                    
        if step == args.test.steps:
            break

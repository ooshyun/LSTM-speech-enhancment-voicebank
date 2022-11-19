"""
 - Reference. 2016, Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech
 
 ITU-T P.56 method [26] to calculate active speech levels using the code provided in [13]. 
 The clean waveforms were added to noise after they had been normalised and silence segments 
 longer than 200 ms had been trimmed off from the beginning and end of each sentence.
"""
from concurrent.futures import ProcessPoolExecutor
from distutils.command.clean import clean
import librosa
import numpy as np
import math
from preprocess.feature_extractor import FeatureExtractor
import multiprocessing
import os
from pathlib import Path
from src.utils import get_tf_feature_mag_phase_pair, get_tf_feature_sample_pair, read_audio, segment_audio
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
import tqdm

class DatasetVoiceBank:
    def __init__(self, clean_filenames, noisy_filenames, name, args):
        self.clean_filenames = clean_filenames
        self.noisy_filenames = noisy_filenames
        self.model_name = name
        self.args = args
        self.debug = True

    def _sample_noisy_filename(self):
        return np.random.choice(self.noisy_filenames)

    def _remove_silent_frames(self, audio, index_indices=None, name=None):
        trimed_audio = []

        if index_indices is None: 
            indices = librosa.effects.split(audio, hop_length=self.args.hop_length, top_db=self.args.top_db) # average mse in each frame < -20 dB
        else:
            indices = index_indices

        audio_remove_slience = np.zeros_like(audio)
        for index in indices:
            audio_remove_slience[index[0]:index[1]] = audio[index[0]: index[1]]
        
        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])

        return indices, np.array(trimed_audio)

    def _phase_aware_scaling(self, clean_spectral_magnitude, clean_phase, noise_phase):
        assert clean_phase.shape == noise_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)

    def audio_process(self, filename):
        clean_filename, noisy_filename = filename
        assert clean_filename.split("/")[-1] == noisy_filename.split("/")[-1], "filename must match."

        clean_audio, sr = read_audio(clean_filename, self.args.sample_rate)
        noisy_audio, sr = read_audio(noisy_filename, self.args.sample_rate)

        eps = 1e-6
        if self.args.normalize == 'z-score':
            clean_audio = (clean_audio-np.mean(clean_audio, axis=-1))/(np.std(clean_audio, axis=-1)+eps)
            noisy_audio = (noisy_audio-np.mean(noisy_audio, axis=-1))/(np.std(noisy_audio, axis=-1)+eps)
        elif self.args.normalize == 'min-max':
            clean_audio = clean_audio/(np.max(clean_audio, axis=-1)-np.min(clean_audio, axis=-1)+eps)
            noisy_audio = noisy_audio/(np.max(clean_audio, axis=-1)-np.min(clean_audio, axis=-1)+eps)
        elif self.args.normalize == 'none':
            pass
        else:
            raise ValueError("Configuration normalization name is incorrect...")

        # # remove silent frame from clean audio
        #     noisy_index, noisy_audio = self._remove_silent_frames(noisy_audio, None, noisy_filename)
        #     noisy_index, clean_audio = self._remove_silent_frames(clean_audio, noisy_index, clean_filename)
            
        # sample random fixed-sized snippets of audio
        clean_audio = segment_audio(clean_audio, self.args.sample_rate, self.args.segment)
        noisy_audio = segment_audio(noisy_audio, self.args.sample_rate, self.args.segment)

        if self.args.fft:
            # extract stft features from noisy audio
            noisy_input_fe = FeatureExtractor(noisy_audio, windowLength=self.args.win_length, hop_length=self.args.hop_length,
                                            sample_rate=self.args.sample_rate)
            noisy_spectrogram = noisy_input_fe.get_stft_spectrogram(self.args.center)

            # Or get the phase angle (in radians)
            # noisy_stft_magnitude, noisy_stft_phase = librosa.magphase(noisy_stft_features)
            noisy_phase = np.angle(noisy_spectrogram)

            # get the magnitude of the spectral
            noisy_magnitude = np.abs(noisy_spectrogram)

            # extract stft features from clean audio
            clean_audio_fe = FeatureExtractor(clean_audio, windowLength=self.args.win_length, hop_length=self.args.hop_length,
                                            sample_rate=self.args.sample_rate)
            clean_spectrogram = clean_audio_fe.get_stft_spectrogram(self.args.center)
            # clean_spectrogram = cleanAudioFE.get_mel_spectrogram()

            # get the clean phase
            clean_phase = np.angle(clean_spectrogram)

            # get the clean spectral magnitude
            clean_magnitude = np.abs(clean_spectrogram)
            # clean_magnitude = 2 * clean_magnitude / np.sum(scipy.signal.hamming(self.args.win_length, sym=False))

            noisy_real, noisy_imag = np.real(noisy_spectrogram), np.imag(noisy_spectrogram)
            clean_real, clean_imag = np.real(clean_spectrogram), np.imag(clean_spectrogram)

            # called phase aware scaling
            # clean_magnitude = self._phase_aware_scaling(clean_magnitude, clean_phase, noisy_phase) 
            # scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
            # noisy_magnitude = scaler.fit_transform(noisy_magnitude)
            # clean_magnitude = scaler.transform(clean_magnitude)

            return noisy_magnitude, clean_magnitude, noisy_phase, clean_phase, noisy_real, clean_real, noisy_imag, clean_imag
        else:
            return noisy_audio, clean_audio


    def create_tf_record(self, *, prefix, subset_size, parallel=False):
        counter = 0
        root = self.args.save_path

        # p = multiprocessing.Pool(multiprocessing.cpu_count())
        print("DEBUG", self.args.fft_normalize)
        if self.debug:
            if self.args.fft_normalize:
                folder = Path(f"{root}/records_{self.model_name}_train_{int(self.args.split*100)}_norm_{self.args.normalize}_fft_{self.args.fft}_norm_topdB_{self.args.top_db}_debug")
            else:
                folder = Path(f"{root}/records_{self.model_name}_train_{int(self.args.split*100)}_norm_{self.args.normalize}_fft_{self.args.fft}_topdB_{self.args.top_db}_debug")
        else:
            if self.args.fft_normalize:
                folder = Path(f"{root}/records_{self.model_name}_train_{int(self.args.split*100)}_norm_{self.args.normalize}_fft_{self.args.fft}_norm_topdB_{self.args.top_db}")
            else:
                folder = Path(f"{root}/records_{self.model_name}_train_{int(self.args.split*100)}_norm_{self.args.normalize}_fft_{self.args.fft}_topdB_{self.args.top_db}")

        if not folder.is_dir():
            folder.mkdir()

        for i in range(0, len(self.clean_filenames), subset_size):
            if self.debug:            
                subset_size = 10

            tfrecord_filename = str(folder / f"{prefix}_{str(counter)}.tfrecords")

            if os.path.isfile(tfrecord_filename):
                print(f"Skipping {tfrecord_filename}")
                counter += 1
                continue

            writer = tf.io.TFRecordWriter(tfrecord_filename)
            file_names_sublist = [(clean_filename, noisy_filename) for clean_filename, noisy_filename in zip(self.clean_filenames[i:i + subset_size], self.noisy_filenames[i:i + subset_size])]
            
            print(f"Processing files from: {i} to {i + subset_size}")
            if parallel: # Didn't work
                print(f"CPU ", os.cpu_count()-3 if os.cpu_count()>4 else 1, "...")
                out = []
                pendings = []
                with ProcessPoolExecutor(os.cpu_count()-3 if os.cpu_count()>4 else 1) as pool:
                    for file_name in file_names_sublist:
                        pendings.append(pool.submit(self.audio_process, file_name))
                    
                    for pending in tqdm.tqdm(pendings):
                        out.append(pending.result())

                # out = p.map(self.parallel_audio_processing, clean_filenames_sublist)
            else:
                out = [self.audio_process(file_names) for file_names in tqdm.tqdm(file_names_sublist, ncols=120)]
            
            for o in out:
                if self.args.fft:
                    noisy_stft_magnitude = o[0]
                    clean_stft_magnitude = o[1]
                    noisy_stft_phase = o[2]
                    clean_stft_phase = o[3]
                    # noisy_stft_real = o[4]
                    # clean_stft_real = o[5]
                    # noisy_stft_imag = o[6]
                    # clean_stft_imag = o[7]
                    if self.debug:
                        print("  Getting from preprocess")
                        print("[DEBUG]: ", noisy_stft_magnitude.shape, noisy_stft_phase.shape, clean_stft_magnitude.shape, clean_stft_phase.shape)
                        print("[DEBUG]: ", noisy_stft_magnitude.dtype, noisy_stft_phase.dtype, clean_stft_magnitude.dtype, clean_stft_phase.dtype)
                        print("---")

                    new_axes = np.arange(len(clean_stft_phase.shape))
                    new_axes[-2:] = new_axes[-1], new_axes[-2]

                    noisy_stft_magnitude = np.transpose(noisy_stft_magnitude, axes=new_axes)
                    clean_stft_magnitude = np.transpose(clean_stft_magnitude, axes=new_axes)
                    noisy_stft_phase = np.transpose(noisy_stft_phase, axes=new_axes)
                    clean_stft_phase = np.transpose(clean_stft_phase, axes=new_axes) # segment, ch, frame, freqeuncy
                    if self.args.fft_normalize:
                        noisy_stft_magnitude /= (self.args.n_feature-1)*2
                        clean_stft_magnitude /= (self.args.n_feature-1)*2 

                    if self.debug:
                        print(" Segmentation")
                        print("[DEBUG]: ", noisy_stft_magnitude.shape, noisy_stft_phase.shape, clean_stft_magnitude.shape, clean_stft_phase.shape)
                        print("[DEBUG]: ", noisy_stft_magnitude.dtype, noisy_stft_phase.dtype, clean_stft_magnitude.dtype, clean_stft_phase.dtype)
                        print("---")

                    for noise_mag, clean_mag, noisy_phase, clean_phase in zip(noisy_stft_magnitude, clean_stft_magnitude, noisy_stft_phase, clean_stft_phase):
                        noise_mag = np.expand_dims(noise_mag, axis=0)  # 1, ch, frame, freqeuncy
                        clean_mag = np.expand_dims(clean_mag, axis=0)
                        noisy_phase = np.expand_dims(noisy_phase, axis=0)
                        clean_phase = np.expand_dims(clean_phase, axis=0)

                        if self.debug:
                            print("  Write Down to tfrecord")
                            print("[DEBUG]: ", noise_mag.shape, noisy_phase.shape, clean_mag.shape, clean_phase.shape)
                            print("[DEBUG]: ", noise_mag.dtype, noisy_phase.dtype, clean_mag.dtype, clean_phase.dtype)
                            print("---")
        
                        example = get_tf_feature_mag_phase_pair(noise_mag, clean_mag, noisy_phase, clean_phase)
                        writer.write(example.SerializeToString())    
                else:
                    noisy_audio = o[0]
                    clean_audio = o[1]
                    
                    if self.debug:
                        print("[DEBUG]: ", noisy_audio.shape, clean_audio.shape)

                    for n, c in zip(noisy_audio, clean_audio):
                        example = get_tf_feature_sample_pair(n, c)
                        writer.write(example.SerializeToString())
            counter += 1
            writer.close()

            if self.debug:            
                break

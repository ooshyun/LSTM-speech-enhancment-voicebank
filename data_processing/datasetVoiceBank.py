from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
import math
from data_processing.feature_extractor import FeatureExtractor
from utils import prepare_input_features
import multiprocessing
import os
from pathlib import Path
from utils import get_tf_feature, read_audio
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
import tqdm
np.random.seed(999)
tf.random.set_seed(999)

model_name = 'lstm'

class DatasetVoiceBank:
    def __init__(self, clean_filenames, noisy_filenames, **config):
        self.clean_filenames = clean_filenames
        self.noisy_filenames = noisy_filenames
        self.sample_rate = config['fs']
        self.overlap = config['overlap']
        self.window_length = config['windowLength']
        self.audio_max_duration = config['audio_max_duration']

    def _sample_noisy_filename(self):
        return np.random.choice(self.noisy_filenames)

    def _remove_silent_frames(self, audio):
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=self.overlap, top_db=20)

        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

    def _phase_aware_scaling(self, clean_spectral_magnitude, clean_phase, noise_phase):
        assert clean_phase.shape == noise_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)

    def get_noisy_audio(self, *, filename):
        return read_audio(filename, self.sample_rate)

    def _audio_random_crop(self, clean_audio, noisy_audio, duration):
        audio_duration_secs = librosa.core.get_duration(clean_audio, self.sample_rate)

        ## duration: length of the cropped audio in seconds
        if duration >= audio_duration_secs:
            # print("Passed duration greater than audio duration of: ", audio_duration_secs)
            return clean_audio, noisy_audio

        audio_duration_ms = math.floor(audio_duration_secs * self.sample_rate)
        duration_ms = math.floor(duration * self.sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return clean_audio[idx: idx + duration_ms], noisy_audio[idx: idx + duration_ms]


    def parallel_audio_processing(self, filename):
        clean_filename, noisy_filename = filename
        assert clean_filename.split("/")[-1] == noisy_filename.split("/")[-1], "filename must match."

        clean_audio, _ = read_audio(clean_filename, self.sample_rate)
        noisy_audio, sr = read_audio(noisy_filename, self.sample_rate)

        # remove silent frame from clean audio
        # clean_audio = self._remove_silent_frames(clean_audio)
        # noisy_audio = self._remove_silent_frames(noisy_audio)

        # sample random fixed-sized snippets of audio
        clean_audio, noisy_audio= self._audio_random_crop(clean_audio, noisy_audio, duration=self.audio_max_duration)

        # extract stft features from noisy audio
        noisy_input_fe = FeatureExtractor(noisy_audio, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        noisy_spectrogram = noisy_input_fe.get_stft_spectrogram()

        # Or get the phase angle (in radians)
        # noisy_stft_magnitude, noisy_stft_phase = librosa.magphase(noisy_stft_features)
        noisy_phase = np.angle(noisy_spectrogram)

        # get the magnitude of the spectral
        noisy_magnitude = np.abs(noisy_spectrogram)

        # extract stft features from clean audio
        clean_audio_fe = FeatureExtractor(clean_audio, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        clean_spectrogram = clean_audio_fe.get_stft_spectrogram()
        # clean_spectrogram = cleanAudioFE.get_mel_spectrogram()

        # get the clean phase
        clean_phase = np.angle(clean_spectrogram)

        # get the clean spectral magnitude
        clean_magnitude = np.abs(clean_spectrogram)
        # clean_magnitude = 2 * clean_magnitude / np.sum(scipy.signal.hamming(self.window_length, sym=False))

        clean_magnitude = self._phase_aware_scaling(clean_magnitude, clean_phase, noisy_phase)

        scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        noisy_magnitude = scaler.fit_transform(noisy_magnitude)
        clean_magnitude = scaler.transform(clean_magnitude)
        return noisy_magnitude, clean_magnitude, noisy_phase

    def create_tf_record(self, *, prefix, subset_size, parallel=False):
        counter = 0
        # p = multiprocessing.Pool(multiprocessing.cpu_count())

        folder = Path(f"./records_{model_name}")
        
        if folder.is_dir():
            pass
        else:
            folder.mkdir()

        for i in range(0, len(self.clean_filenames), subset_size):

            tfrecord_filename = str(folder / f"{prefix}_{str(counter)}.tfrecords")

            if os.path.isfile(tfrecord_filename):
                print(f"Skipping {tfrecord_filename}")
                counter += 1
                continue

            writer = tf.io.TFRecordWriter(tfrecord_filename)
            
            # clean_filenames_sublist = self.clean_filenames[i:i + subset_size]
            # noisy_filenames_sublist = self.noisy_filenames[i:i + subset_size]

            file_names_sublist = [(clean_filename, noisy_filename) for clean_filename, noisy_filename in zip(self.clean_filenames[i:i + subset_size], self.noisy_filenames[i:i + subset_size])]
            
            print(f"Processing files from: {i} to {i + subset_size}")
            if parallel: # Didn't work
                print(f"CPU ", os.cpu_count()-3 if os.cpu_count()>4 else 1, "...")
                out = []
                pendings = []
                with ProcessPoolExecutor(os.cpu_count()-3 if os.cpu_count()>4 else 1) as pool:
                    for file_name in file_names_sublist:
                        pendings.append(pool.submit(self.parallel_audio_processing, file_name))
                    
                    for pending in tqdm.tqdm(pendings):
                        out.append(pending.result())

                # out = p.map(self.parallel_audio_processing, clean_filenames_sublist)
            else:
                out = [self.parallel_audio_processing(file_names) for file_names in tqdm.tqdm(file_names_sublist, ncols=120)] 
            
            for o in out:
                noisy_stft_magnitude = o[0]
                clean_stft_magnitude = o[1]
                noisy_stft_phase = o[2]

                if model_name == "cnn":
                    # cnn-denoiser input, 8 segementation, 256 window, num frequency, num channel, num segment, 
                    noise_stft_mag_features = prepare_input_features(noisy_stft_magnitude, numSegments=8, numFeatures=129) # cnn-denoiser

                    noise_stft_mag_features = np.transpose(noise_stft_mag_features, (2, 0, 1)) # nchannel, nseg, nfeature -> nfeature, nchannel, nseg
                    clean_stft_magnitude = np.transpose(clean_stft_magnitude, (1, 0))
                    noisy_stft_phase = np.transpose(noisy_stft_phase, (1, 0))

                    noise_stft_mag_features = np.expand_dims(noise_stft_mag_features, axis=3)
                    clean_stft_magnitude = np.expand_dims(clean_stft_magnitude, axis=2)

                    for x_, y_, p_ in zip(noise_stft_mag_features, clean_stft_magnitude, noisy_stft_phase):
                        y_ = np.expand_dims(y_, 2)
                        example = get_tf_feature(x_, y_, p_)
                        writer.write(example.SerializeToString())

                    # lstm, 1 sec segementation, 512 window, num channel, num segment, num frequency
                    noise_stft_mag_features = prepare_input_features(noisy_stft_magnitude, numSegments=8, numFeatures=256) # already segmentation in 1 sec
                
                elif model_name == "lstm":
                    noisy_stft_magnitude = np.transpose(noisy_stft_magnitude, (1, 0))
                    clean_stft_magnitude = np.transpose(clean_stft_magnitude, (1, 0))
                    noisy_stft_phase = np.transpose(noisy_stft_phase, (1, 0))

                    noisy_stft_magnitude = np.expand_dims(noisy_stft_magnitude, axis=0)
                    clean_stft_magnitude = np.expand_dims(clean_stft_magnitude, axis=0)
                    noisy_stft_phase = np.expand_dims(noisy_stft_phase, axis=0)

                    for x_, y_, p_ in zip(noisy_stft_magnitude, clean_stft_magnitude, noisy_stft_phase):
                        example = get_tf_feature(x_, y_, p_)
                        writer.write(example.SerializeToString())    
                else:
                    logging.info("Since not implemented model, so no processing...")
                    continue
                
            counter += 1
            writer.close()

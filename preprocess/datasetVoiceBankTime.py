from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
import math
import multiprocessing
import os
from pathlib import Path
from model.utils import get_tf_feature, read_audio, get_tf_feature_time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
import tqdm
np.random.seed(999)
tf.random.set_seed(999)

class DatasetVoiceBankTime:
    def __init__(self, clean_filenames, noisy_filenames, name, args):
        self.clean_filenames = clean_filenames
        self.noisy_filenames = noisy_filenames
        self.model_name = name
        self.sample_rate = args.sample_rate
        self.overlap = args.hop_length
        self.window_length = args.win_length
        self.audio_max_duration = args.segment
        self.top_db = args.top_db
        self.max_db = args.max_db
        self.save_path = args.save_path
        self.domain = args.domain
        self.debug = False

    def _sample_noisy_filename(self):
        return np.random.choice(self.noisy_filenames)

    def _remove_silent_frames(self, audio, index_indices=None, name=None):
        trimed_audio = []

        if index_indices is None: 
            indices = librosa.effects.split(audio, hop_length=self.overlap, top_db=self.top_db) # average mse in each frame < -20 dB
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

        if self.top_db <= self.max_db:
            noisy_index, noisy_audio = self._remove_silent_frames(noisy_audio, None, noisy_filename)
            noisy_index, clean_audio = self._remove_silent_frames(clean_audio, noisy_index, clean_filename)

        # sample random fixed-sized snippets of audio
        clean_audio, noisy_audio= self._audio_random_crop(clean_audio, noisy_audio, duration=self.audio_max_duration)

        if len(clean_audio.shape) == 1:
            noisy_audio = np.expand_dims(noisy_audio, axis=0)
            clean_audio = np.expand_dims(clean_audio, axis=0)

        return noisy_audio, clean_audio

    def create_tf_record(self, *, prefix, subset_size, parallel=False):
        counter = 0
        root = self.save_path
        # p = multiprocessing.Pool(multiprocessing.cpu_count())
        if self.debug:
            folder = Path(f"{root}/records_{self.model_name}_{self.domain}_{self.top_db}topdb_debug")
        else:
            if self.top_db <= self.max_db:
                folder = Path(f"{root}/records_{self.model_name}_{self.domain}_{self.top_db}topdb")
            else:
                folder = Path(f"{root}/records_{self.model_name}_{self.domain}")
            
        if folder.is_dir():
            pass
        else:
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
                noisy = o[0]
                clean = o[1]
                
                if self.debug:
                    print("[DEBUG]: ", noisy.shape, clean.shape)

                for n, c in zip(noisy, clean):
                    example = get_tf_feature_time(n, c)
                    writer.write(example.SerializeToString())    
                else:
                    logging.info("Since not implemented model, so no processing...")
                    continue
                
            counter += 1
            writer.close()

            if self.debug:            
                break # Test
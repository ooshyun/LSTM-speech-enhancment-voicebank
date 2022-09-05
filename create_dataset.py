from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from data_processing.urban_sound_8K import UrbanSound8K
from data_processing.VoiceBankDEMAND import VoiceBandDEMAND
from data_processing.dataset import Dataset
from data_processing.datasetVoiceBank import DatasetVoiceBank
import warnings

warnings.filterwarnings(action='ignore')

# mozilla_basepath = '/home/thallessilva/Documents/datasets/en'
# urbansound_basepath = '/home/thallessilva/Documents/datasets/UrbanSound8K'

# mcv = MozillaCommonVoiceDataset(mozilla_basepath, val_dataset_size=1000)
# clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

# us8K = UrbanSound8K(urbansound_basepath, val_dataset_size=200)
# noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()

voiceBankDEMAND_basepath = '/Users/seunghyunoh/workplace/study/NoiseReduction/Tiny-SpeechEnhancement/data/VoiceBankDEMAND/DS_10283_2791'
voiceBankDEMAND_basepath = '/home/daniel0413/workplace/project/SpeechEnhancement/TinyML/data/VoiceBankDEMAND'


voiceBank = VoiceBandDEMAND(voiceBankDEMAND_basepath, val_dataset_percent=0.3)
clean_train_filenames, noisy_train_filenames, clean_val_filenames, noisy_val_filenames = voiceBank.get_train_val_filenames()

# # cnn-denoiser
# windowLength = 256

# config = {'windowLength': windowLength,
#           'overlap': round(0.25 * windowLength),
#           'fs': 16000,
#           'audio_max_duration': 0.8}

# val_dataset = DatasetVoiceBank(clean_val_filenames, noisy_val_filenames, **config)
# val_dataset.create_tf_record(prefix='val', subset_size=2000)

# train_dataset = DatasetVoiceBank(clean_train_filenames, noisy_train_filenames, **config)
# train_dataset.create_tf_record(prefix='train', subset_size=4000)

## Create Test Set
# clean_test_filenames = mcv.get_test_filenames()
# noise_test_filenames = us8K.get_test_filenames()

# lstm
windowLength = 512

config = {'windowLength': windowLength,
          'overlap': round(0.5 * windowLength),
          'fs': 16000,
          'audio_max_duration': 1}

val_dataset = DatasetVoiceBank(clean_val_filenames, noisy_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=2000)

train_dataset = DatasetVoiceBank(clean_train_filenames, noisy_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=4000)

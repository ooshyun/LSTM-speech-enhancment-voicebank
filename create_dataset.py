from preprocess.VoiceBankDEMAND import VoiceBandDEMAND
from preprocess.dataset import DatasetVoiceBank
import os
import pickle
import warnings
from src.utils import load_yaml

warnings.filterwarnings(action='ignore')

def preprocess_data(args):
    file_name = f"voicebank_dataset_list_train_{int(args.dset.split*100)}_val_{int(100-args.dset.split*100)}.pkl"
    data_path = args.dset.wav
    
    if not os.path.exists(args.dset.save_path):
        os.mkdir(args.dset.save_path)

    file_list_pkl = os.path.join(args.dset.save_path, file_name)
    if os.path.exists(file_list_pkl):
        with open(file_list_pkl, 'rb') as tmp:
            clean_train_filenames, noisy_train_filenames, clean_val_filenames, noisy_val_filenames = pickle.load(tmp)
    else:
        dataset_voicebank = VoiceBandDEMAND(data_path, val_dataset_percent=1-args.dset.split)
        clean_train_filenames, noisy_train_filenames, clean_val_filenames, noisy_val_filenames = dataset_voicebank.get_train_val_filenames()
        with open(file_list_pkl, 'wb') as tmp:
            pickle.dump([clean_train_filenames, noisy_train_filenames, clean_val_filenames, noisy_val_filenames], tmp)

    if args.model.name == "lstm":
        if args.dset.fft:
            train_dataset = DatasetVoiceBank(clean_train_filenames, noisy_train_filenames, args.model.name, args.dset, args.debug)
            train_dataset.create_tf_record(prefix='train')
            val_dataset = DatasetVoiceBank(clean_val_filenames, noisy_val_filenames, args.model.name, args.dset, args.debug)
            val_dataset.create_tf_record(prefix='val')
        else:
            train_dataset = DatasetVoiceBank(clean_train_filenames, noisy_train_filenames, args.model.name, args.dset, args.debug)
            train_dataset.create_tf_record(prefix='train')
            val_dataset = DatasetVoiceBank(clean_val_filenames, noisy_val_filenames, args.model.name, args.dset, args.debug)
            val_dataset.create_tf_record(prefix='val')
    else:
        raise NotImplementedError("There's no implementation, ", args.model.name)

if __name__=="__main__":
  path_conf = "./conf/config.yaml"
  config = load_yaml(path_conf)
  preprocess_data(config)

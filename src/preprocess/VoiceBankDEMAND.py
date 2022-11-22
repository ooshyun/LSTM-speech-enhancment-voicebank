import glob
import numpy as np
from pathlib import Path
import os
import copy

EXT_LIST = ['wav']

def _find_files(path_list):
    file_list = []

    for path in path_list:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                    continue
            
            file_names = [str(root/file) for file in files if str(file.split(".")[-1]) in EXT_LIST]
            file_list += copy.deepcopy(file_names)
            del file_names
    
    file_list = sorted(file_list)

    return file_list

class VoiceBandDEMAND:
    def __init__(self, basepath, *, val_dataset_percent, class_ids=None):
        self.basepath = basepath
        self.val_dataset_percent = val_dataset_percent
        self.class_ids = class_ids

    def _get_filenames(self, split='train'):
        file_clean_list, file_noisy_list = None, None
        
        if split == 'train':
            folder_clean_datasets = glob.glob(os.path.join(self.basepath, '*clean_train*'))
            folder_noisy_datasets = glob.glob(os.path.join(self.basepath, '*noisy_train*'))
            file_clean_list = _find_files(folder_clean_datasets)
            file_noisy_list = _find_files(folder_noisy_datasets)
        else:
            folder_clean_datasets = glob.glob(os.path.join(self.basepath, '*clean_test*'))
            folder_noisy_datasets = glob.glob(os.path.join(self.basepath, '*noisy_test*'))
            file_clean_list = _find_files(folder_clean_datasets)
            file_noisy_list = _find_files(folder_noisy_datasets)

        print("File example:")
        print("Clean: ", file_clean_list[0], "The number: ", len(file_clean_list))
        print("Noisy: ", file_noisy_list[0], "The number: ", len(file_noisy_list))

        return file_clean_list, file_noisy_list


    def get_train_val_filenames(self):
        voicebank_filenames_clean, voicebank_filenames_noisy= self._get_filenames()
        voicebank_id = np.arange(len(voicebank_filenames_clean))
        np.random.shuffle(voicebank_id)

        # separate noise files for train/validation
        len_val = int(len(voicebank_id)*self.val_dataset_percent)
        voicebank_train_clean_shuffle = [voicebank_filenames_clean[id] for id in voicebank_id[:-len_val]] 
        voicebank_train_noisy_shuffle = [voicebank_filenames_noisy[id] for id in voicebank_id[:-len_val]] 

        voicebank_val_clean_shuffle = [voicebank_filenames_clean[id] for id in voicebank_id[-len_val:]]
        voicebank_val_noisy_shuffle = [voicebank_filenames_noisy[id] for id in voicebank_id[-len_val:]]

        print("Training:", len(voicebank_train_clean_shuffle))
        print("Validation:", len(voicebank_val_clean_shuffle))

        return voicebank_train_clean_shuffle, voicebank_train_noisy_shuffle, voicebank_val_clean_shuffle, voicebank_val_noisy_shuffle, 

    def get_test_filenames(self):
        voicebank_filenames_clean, voicebank_filenames_noisy = self._get_filenames('test')
        voicebank_id = np.arange(len(voicebank_filenames_clean))
        np.random.shuffle(voicebank_id)
        voicebank_filenames_clean_shuffle = [voicebank_filenames_clean[id] for id in voicebank_id] 
        voicebank_filenames_noisy_shuffle = [voicebank_filenames_noisy[id] for id in voicebank_id] 

        del voicebank_id, voicebank_filenames_clean, voicebank_filenames_noisy

        print("# of Noise testing files:", len(voicebank_filenames_clean_shuffle))
        return voicebank_filenames_clean_shuffle, voicebank_filenames_noisy_shuffle

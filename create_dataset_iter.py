from src.utils import load_yaml, limit_gpu_tf
from create_dataset import preprocess_data

if __name__ == "__main__":
    limit_gpu_tf(768)

    path_conf = "./conf/config_preprocess.yaml"
    config = load_yaml(path_conf)

    fft_list = (True, False)
    split_list = (0.9, 0.8, 0.7)
    normalize_list = ("min-max", "z-score", "none")
    segment_normalization_list = (True, False)

    config.dset.fft = True
    for split in split_list:
        for segment_normalization in segment_normalization_list:
            for normalize in normalize_list:
                    config.dset.split = split
                    config.dset.segment_normalization = segment_normalization
                    config.dset.normalize = normalize
                    preprocess_data(config)

    config.dset.fft_normalize = False
    config.dset.fft = False
    for split in split_list:
        for segment_normalization in segment_normalization_list:
            for normalize in normalize_list:
                config.dset.segment_normalization = segment_normalization
                config.dset.normalize = normalize
                preprocess_data(config)

# LSTM model for Speech Enhancment

This repository is forked form [A Fully Convolutional Neural Network for Speech Enhancement](https://github.com/EncoraDigital/SAB-cnn-audio-denoiser), but the contents is different.
Previously, the cnn-audio repository includes dataset MozillaCommonVoice and UrbanSound. In my case, I cannot access previous dataset in the reason I am not in academic. Therefore, we modify the entire structure for VoiceBankDEMAND dataset. In additions, this repository didn't include previous model(CNN), because my final goal is implementing this model to tiny devices. Hoewever, there's not lots of open-sources for implementing ML model to embedded devices, especially for speech enhancement. So it will approach LSTM model to implement on the device named STM32F746, which was referred by [Speech enhancment made by Bose](https://github.com/Bose/efficient-neural-speech-enhancement). 

This includes tensorflow 2.9 and tensorflow api such as fit, callback, but this has limitation for implementing custom loss function. For example, if I want to compute loss referring the metadata(ex. mean, std) during training, the tensorflow only conveys estimation, true data, and weight. Even if I tried to convey the data including those metadata, it will be wasteful approach. The metrics such as stoi, sdr, pesq also has [some issues]() to implement. So this repository is for understanding how tensorflow framework operate when they're training the model, and especially you can observe the model will have convergence. And the other work such as implementation to tiny devices continues to the [other repository](https://github.com/ooshyun/TinyLSTM-for-speech-enhancement).

## Details

### 1. Tree
---
Tree of this repository is as below
```
    .
    ├── README.md
    ├── conf
    │   ├── config.yaml
    │   └── config_preprocess.yaml
    ├── create_dataset.py
    ├── create_dataset_iter.py
    ├── data
    ├── history
    │   └── etc
    ├── inference.ipynb
    ├── result
    │   └── lstm
    ├── src
    │   ├── __init__.py
    │   ├── distrib.py
    │   ├── lstm.py
    │   ├── metrics.py
    │   ├── preprocess
    │   │   ├── VoiceBankDEMAND.py
    │   │   ├── __init__.py
    │   │   ├── dataset.py
    │   │   └── feature_extractor.py
    │   └── utils.py
    ├── test
    │   ├── __init__.py
    │   ├── analyze_dataset.ipynb
    │   ├── conf
    │   │   ├── config.yaml
    │   │   └── config_init.yaml
    │   ├── result
    │   ├── test_dataset.py
    │   └── test_model.py
    └── train.py 
```
The repository consitute configuration, create dataset after preprocess, train, inference, and test of apis. 

### 2. Configuration
Configuration includes several features. The list is as below.

fft size
window size
hop length
fft data or samples
normalization in wav samples
normalization in segmentation
remove slience

Dataset related with model 
the number of feature 
hop length
the number of frames in fft for segmentation
the number of mel-spectrogram
minimum frequency in mel-spectrogram
maximum frequency in mel-spectrogram
option of stft for streaming data


### 3. Preprocess


### 4. Train


### 5. Inference


## Reference
- Train and Validate: https://keras.io/examples/audio/transformer_asr/
- Mel: https://keras.io/examples/audio/melgan_spectrogram_inversion/

- conf: configuration file
- history: previous files
- preprocess: preprocess dataset
- result: saved model
- logs: tensorboard
- test: test functionality


[TODO]
- library
    - scipy 1.8.1
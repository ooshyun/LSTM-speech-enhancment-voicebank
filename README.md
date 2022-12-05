# LSTM model for Speech Enhancment

This repository is forked form [A Fully Convolutional Neural Network for Speech Enhancement](https://github.com/EncoraDigital/SAB-cnn-audio-denoiser), but the contents is different.
Previously, the cnn-audio repository includes dataset MozillaCommonVoice and UrbanSound. In my case, I cannot access previous dataset in the reason I am not in academic. Therefore, we modify the entire structure for VoiceBankDEMAND dataset. In additions, this repository didn't include previous model(CNN), because my final goal is implementing this model to tiny devices. Hoewever, there's not lots of open-sources for implementing ML model to embedded devices, especially for speech enhancement. So it will approach LSTM model to implement on the device named STM32F746, which was referred by [Speech enhancment made by Bose](https://github.com/Bose/efficient-neural-speech-enhancement). 

This includes tensorflow 2.9 and tensorflow api such as fit, callback, but this has limitation for implementing custom loss function. For example, if I want to compute loss referring the metadata(ex. mean, std) during training, the tensorflow only conveys estimation, true data, and weight. Even if I tried to convey the data including those metadata, it will be wasteful approach. The metrics such as stoi, sdr, pesq also has [some issues](https://github.com/ooshyun/LSTM-speech-enhancment-voicebank/blob/master/src/lstm.py) to implement. (If you want, watch SpeechMetric from [this link](https://github.com/ooshyun/LSTM-speech-enhancment-voicebank/blob/master/src/lstm.py).) So this repository is for understanding how tensorflow framework operate when they're training the model, and especially you can observe the model will have convergence. And the other work such as implementation to tiny devices continues to the [other repository](https://github.com/ooshyun/TinyLSTM-for-speech-enhancement).

## Details
### 0. Before the Start
The environment is based on MacOS and Linux with miniconda, which I tested. Using *.yml files, then run the conda environment.

- make conda environment            : conda env create "environment name"
- make conda environment with yml   : conda env create -f environment-macos.yml
- export conda environment          : conda env export > environment.yml

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
The repository consitute configuration, create dataset after preprocess, train, inference, and test of apis. After this line, each of parts explains the major part of this repository and structure.

### 2. Configuration

1. Dataset: dset
    ```
    wav                     : path for dataset
    split                   : ratio between train and validation
    sample_rate             : resample rate in preprocess
    segment                 : the length of segmentation
    n_fft                   : fft size
    win_length              : window size
    hop_length              : hop size
    channels                : channel in preprocess, but not used
    top_db                  : remove slience
    fft                     : fft data(True) or samples(False)
    center                  : option of stft for streaming data (True: no, False: yes)
    save_path               : path for saving preprocess dataset
    fft_normalize           : True
    normalize               : normalization in wav samples
    segment_normalization   : normalization in segmentation(True: yes, False: no) 
    ```
2. Model: model
    ```
    name                    : the type of model 
    lstm_layer              : the number of LSTM layer
    n_feature               : the number of features
    # 1.008 sec in 512 window, 256 hop, sr = 16000 Hz
    n_segment               : the number of frames in fft for segmentation 
    n_mels                  : the number of mel-spectrogram
    f_min                   : minimum frequency in mel-spectrogram
    f_max                   : maximum frequency in mel-spectrogram
    metric                  : speech related objective metrics, currently only using 'si-sdr'
    path                    : path for loading trained-model
    ckpt                    : path for loading checkpoint of trained-model
    ```

- Others
    ```
    optim
    seed
    batch_size
    steps
    epochs
    folder
    debug
    ```

### 3. Preprocess, create_dataset.py
```
1. Load the configuration

2. Find the files and Save the lists as the form, pickle

3. Split the train and validation files after mixing the file list randomly

4. Process
    1) load the wav file
    2) normalize(z-score, linear method)
    3) [Currently, commented] remove silent frame from clean audio
    4) segment the wav files
    5) short time fourier transform in librosa
    6) pass amplitude, phase, real, imag
    7) save as the form, tfrecord
```

### 4. Train, train.py
```
1. Load the configuration

2. Load dataset
    - load filenames with tfrecord
    - shuffle
    - load dataset using TFRecordDataset
    - shuffle, repeat, batch, prefetch, tf.data.experimental.ignore_errors

3. Load model
    - build model
    - load weight of model
    - compilde the model with optimizer and metrics
    
4. Load callback for tensorflow
    - Tensorboard
    - Checkpoint
    - Early Stopping
    - Time history

5. Train using fit in tensorflow

6. Save model and Optimizer

7. Save configuration 
```

### 5. Inference, inference.ipynb
```
1. Load trained model and configuration from saved folder
    - Refer the path
    - Load the configuration
    - Load/build/compile the model

2. Process test wav file
    - load wav file
    - normalize wav
    - convert stft
    - normalize fft by fft size

3. Apply model
    - reshape the stft data
    - apply model
    - revert to audio
        - revert from amplitude and phase to stft
        - revert stft from stft normalized on a frame
        - istft
        - revert audio sample from istft
        - apply meta data from normalization in audio samples

4. Show the result
    - plot, ipd.Audio, stft
```

### 6. Result
The results links to [this](https://github.com/ooshyun/LSTM-speech-enhancment-voicebank/blob/dev/history/221122-1127/README.md).

## Reference
- Train and Validate: https://keras.io/examples/audio/transformer_asr/
- Mel: https://keras.io/examples/audio/melgan_spectrogram_inversion/

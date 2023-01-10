# ML models for Speech Enhancment

This repository is forked form [A Fully Convolutional Neural Network for Speech Enhancement](https://github.com/EncoraDigital/SAB-cnn-audio-denoiser), but the contents is different.
Previously, the cnn-audio repository includes dataset MozillaCommonVoice and UrbanSound. In my case, I cannot access previous dataset in the reason I am not in academic. Therefore, we modify the entire structure for VoiceBankDEMAND dataset. In additions, this repository didn't include previous model(CNN), because my final goal is implementing this model to tiny devices. Hoewever, there's not lots of open-sources for implementing ML model to embedded devices, especially for speech enhancement. So it will approach LSTM model to implement on the device named STM32F746, which was referred by [Speech enhancment made by Bose](https://github.com/Bose/efficient-neural-speech-enhancement). 

This includes tensorflow 2.9 and tensorflow api such as fit, callback, but this has limitation for implementing custom loss function. For example, if I want to compute loss referring the metadata(ex. mean, std) during training, the tensorflow only conveys estimation, true data, and weight. Even if I tried to convey the data including those metadata, it will be wasteful approach. The metrics such as stoi, sdr, pesq also has [some issues](https://github.com/ooshyun/LSTM-speech-enhancment-voicebank/blob/master/src/lstm.py) to implement. (If you want, watch SpeechMetric from [this link](https://github.com/ooshyun/LSTM-speech-enhancment-voicebank/blob/master/src/lstm.py).) So this repository is for understanding how tensorflow framework operate when they're training the model, and especially you can observe the model will have convergence. And the other work such as implementation to tiny devices continues to the [other repository](https://github.com/ooshyun/TinyLSTM-for-speech-enhancement).

## 0. Before the Start
The environment is based on MacOS and Linux with miniconda, which I tested. Using *.yml files, then run the conda environment.

- make conda environment            : conda env create "environment name"
- make conda environment with yml   : conda env create -f environment-macos.yml
- export conda environment          : conda env export > environment.yml

## 1. Tree
---
Tree of this repository is as below
```
    .
    ├── README.md
    ├── data
    ├── conf
    │   └── config.yaml
    ├── history
    │   └── etc
    ├── result
    ├── src
    │   ├── model
    │   │   ├── __init__.py
    │   │   ├── loss.py
    │   │   ├── metric.py
    │   │   ├── time_frequency.py
    │   │   ├── rnn.py
    │   │   ├── unet.py
    │   │   └── crn.py
    │   ├── preprocess
    │   │   ├── __init__.py
    │   │   ├── dataset.py
    │   │   ├── feature_extractor.py
    │   │   └── VoiceBankDEMAND.py
    │   ├── __init__.py
    │   ├── create_dataset.py
    │   ├── train.py
    │   ├── inference.py
    │   ├── convert_tflite.py
    │   ├── distrib.py
    │   └── utils.py
    ├── test
    │   ├── conf
    │   │   └── config.yaml
    │   ├── __init__.py
    │   ├── analyze_dataset.ipynb
    │   ├── result
    │   ├── test_dataset.py
    │   └── test_model.py
    ├── inference.ipynb
    └── main.py 
```
The repository consitute configuration, code (preprocess, model, train, inference, tflite for quantized model) and test of functions. After this line, each of parts explains the major part of this repository and structure.

## 2. Configuration

### 2.1. Dataset: dset

This configuration is for preprocess of dataset.

```
wav                     : path for dataset
split                   : ratio between train and validation
sample_rate             : resample rate in preprocess
segment                 : the length of segmentation(sec)
n_fft                   : fft size
win_length              : window size
hop_length              : hop size
channels                : channel in preprocess, but not used
top_db                  : remove slience
fft                     : fft data(True) or samples(False)
center                  : option of stft for streaming data (True: no, False: yes)
save_path               : path for saving preprocess dataset
normalize               : normalization in wav samples
segment_normalization   : normalization in segmentation(True: yes, False: no) 
```

### 2.2. Model: model

This configuration is for model.

- model: 'rnn', 'lstm', 'gru', 'crn', 'unet'
- metric: 'sisdr', 
- [TODO] metric: 'nb-pesq', 'sdr', 'stoi', 'wb-pesq'

```
name                    : the type of model 
lstm_layer              : the number of recurrent type model layer(RNN, LSTM, GRU)
n_feature               : the number of features
n_segment               : the number of frames in fft for segmentation 
n_mels                  : the number of mel-spectrogram
f_min                   : minimum frequency in mel-spectrogram
f_max                   : maximum frequency in mel-spectrogram
metric                  : speech related objective metrics, currently only using 'si-sdr'
path                    : path for loading trained-model
ckpt                    : path for loading checkpoint of trained-model
fft_normalization       : bool for normalization of fft
```

### 2.3. Test: test

This configuration is for testing data or inference

```
steps                   : the number of data
wav                     : the path for test data
save                    : bool for saving enhanced results
```
4. TFlite

This configuration is for converting TFLite model

```
format                  : quantization format, int8 or float32
test                    : bool for getting enhanced data [TODO]
```

5. Optimizer

This configuration is for optimizer setting

- The type of Loss: 'mse', 'rmse', 'ideal-mag', 'psa', 'psa-bose'

```
load                    : bool to load optimizer when model load
optim                   : the type of optimizer
lr                      : learning rate
loss                    : the type of loss
```

6. Others

```
seed
batch_size
steps
epochs
folder
debug
```

## 3. Detail of each steps
### 3.1 Preprocess, create_dataset.py
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

### 3.2. Train, train.py
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

### 3.3. Inference, inference.ipynb
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
The results links to [this](./history/221122-1127/README.md). This had only results currently.

## Reference
- Train and Validate: https://keras.io/examples/audio/transformer_asr/
- Mel: https://keras.io/examples/audio/melgan_spectrogram_inversion/

# Test LSTM Model in speech enhancment task

This is comparision using LSTM model in Speech Enhancement task. It used several types of variables to confirm the baseline.

- Loss function, Split ratio, train vs valid, Dense layer initial method, Time Period of Normalization,batch size, Batch Normalization, FFT size, LSTM Layer, Hop size, Hop size / LSTM size

Those variable is the basic component when we design the model. 

[TL;DR] The difference between variable is not too big, but the most performance setting is as below,

- Loss function     : phase sensitive approximate
- Split ratio       : 90:10
- Dense layer initial method : almost same
- Time Period of Normalization: entire
- Batch size: 512
- Batch Normalization: O
- FFT size: 512
- LSTM Layer: 256
- Hop size: 128
- Hop size / LSTM size: 128 / 128

Each of components can be adjusted to reduce training time and the gap between training and validation for confirming performance. This results is because when recovering STFT to time samples for evaluation, the scaling factor is not included, and it effects the difference. However, the sound for testing is quite better than 50% overlap. This project is contiuning to implementation in embedded device. So it will be on [--](). The detail comparison is [this link](). And, I attached details as belwo also.

## Variable
- Default setting
- 20221123-183326
    - Total dataset: 75805 8444
    - Train 90, Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss: psa
    - Dense: glorot_uniform
    - batch: 512
    - lstm layer: 256
    - Batch Normalization after lstm layer
    - metric: loss, sisdr
    - every epoch consists of 1000 step filled with dataset
        - If 2048 samples is divided by 64/256 batch, dataset can be divided as 8, 4 case.
        - Then it will filled until 1000 steps

    - After "preprocess include stft" and "process stft during training", it keeps to use time sample data, not stft

- What is comparison?
    - The shape of Convex(.png)
    - Training and Validation of SISDR and Loss at Convex
    - time in each epoch 
    - total time/the number of iternation until convex
    - the difference between train and validation
    
- Not include normalization method, z-score is the best performance in SISDR

- loss
    - rmse of stft      : 20221122-150347
    - mse of stft       : 20221123-093354
    - psa               : 20221123-183326
    - rmsa of amplitude : 20221124-105217

- Split ratio, train vs valid
    - 90:10: 20221123-183326
    - 80:20: 20221124-181145
    - 70:30: 20221125-094742
    - 50:50: 20221125-171307

- Dense layer initial method
    - uniform   :  20221123-183326
    - normal    :  20221126-091953

- Time Period of Normalization 
    - normalize the wav form by entire wav samples
        20221123-183326
    - normalize the wav form by   1sec wav samples
        20221126-195423

- batch size
    - 512       : 20221123-183326
    - 64        : 20221127-081443

- Preprocess method
    - stft: 20221123-183326
    - time: 20221127-131036

- Batch Norm 
    - batch norm O : 20221123-183326
    - batch norm X : 20221127-220836

- FFT size
    - 512: 20221123-183326
    - 256: 20221128-081941

- LSTM Layer
    - 256: 20221123-183326
    - 128: 20221128-151953 - 20221129-072205

- Hop size
    - 256: 20221123-183326
    - 128: 20221201-190504 - 20221202-132108

- Hop size / LSTM size
    - 256/256: 20221123-183326
    - 128/128: 20221129-115620 - 20221130-144932

## History

- 20221122-150347, 14.96 / 14.76
    - Train 90, Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss: rmse, stft
    - Dense: glorot_uniform
    - batch: 512
    
- 20221123-093354, 14.50, 14.41
    - Train 90, Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss mse, stft
    - Dense: glorot_uniform
    - batch: 512

- 20221123-183326, 14.9814 / 14.7328
    - Train 90, Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform
    - batch: 512

- 20221124-105217, 14.8656  / 14.6824
    - Train 90, Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss rmse, amplitude
    - Dense: glorot_uniform
    - batch: 512

- 20221124-181145, 14.9484, 14.7106
    - Train 80, Validation 20
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform			
    - batch: 512

- 202211225-094742, 14.9695, 14.6493
    - Train 70, Validation 30
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform	
    - batch: 512

- 202211225-171307, 14.9049, 14.6324
    - Train 50, Validation 50
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform				
    - batch: 512

- 20221126-091953, 14.9582, 14.7428
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_normal
    - batch: 512

- 20221126-153602, 
    - Node: model/dense_1/Tensordot/Reshape
        Size 1 must be negative, not -1782199552
        ...

    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: True
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform
    - batch: 512

- 20221126-195423, 14.9347, 14.6850
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: True
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform
    - batch: 512

- 20221127-081443, 14.8441, 14.6418
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - loss psa
    - Dense: glorot_uniform
    - batch: 64

- 20221127-131036, 14.9657, 14.7343
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - Time
    - loss psa
    - Dense: glorot_uniform
    - batch: 512

- 20221127-220836, 14.9, 14.66
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - Time
    - loss psa
    - Dense: glorot_uniform
    - batch: 512
    - Batch Noramlization X

- 20221128-081941, 13.48, 13.31
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - Time
    - fft 256
    - hop size 128
    - lstm layer 256
    - loss psa
    - Dense: glorot_uniform
    - batch: 512
    - Batch Noramlization O

- 20221128-151953 - 20221129-072205, 14.94, 14.77
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - Time
    - fft 512
    - hop size 256
    - lstm layer 128
    - loss psa 
    - Dense: glorot_uniform
    - batch: 512
    - Batch Noramlization O		

- 20221129-115620 - 20221130-144932, 15.24, 15.11
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - Time
    - fft 512
    - hop size 128
    - lstm layer 128
    - loss psa 
    - Dense: glorot_uniform
    - batch: 512
    - Batch Noramlization O	

- 20221201-190504 - 20221202-132108, 15.28, 15.03
    - Train 90 Validation 10
    - Normalization: z-score
    - Segmentation Norm: False
    - FFT normalized by fft size
    - Time
    - fft 512
    - hop size 128
    - lstm layer 256
    - loss psa 
    - Dense: glorot_uniform
    - batch: 512
    - Batch Noramlization O	

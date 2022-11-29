# Test LSTM Model in speech enhancment task

## Variable
- Default setting, 20221123-183326
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
    - [TODO] Test of SISDR and Loss at Convex
    
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

# A Fully Convolutional Neural Network for Speech Enhancement

Tensorflow 2.0 implementation of the paper [A Fully Convolutional Neural Network for Speech Enhancement](https://pdfs.semanticscholar.org/9ed8/e2f6c338f4e0d1ab0d8e6ab8b836ea66ae95.pdf)

Blog post: [Practical Deep Learning Audio Denoising](https://medium.com/better-programming/practical-deep-learning-audio-denoising-79c1c1aea299)

## Dataset

Part of the dataset used to train the original system is now [available to download](http://cdn.daitan.com/dataset.zip)
  
The zip file contains 1 training file (that is 10% of the data used to train the system), a validation file, and two 
audio files (not included in the training files) used to evaluate the model. 

You can create the dataset for yourself. 

- Download the [Mozilla Common Voice](https://voice.mozilla.org/) and [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) datasets.
- Use the ```create_dataset.py``` script to create the TFRecord files. 


- TODO
- mel_filterbank needs the square of amplitude?
- After saving the model, warning is as below
WARNING:absl:Found untraced functions such as lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_1_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.

- Why the num of CNN eval data 4000 and the num of LSTM eval data 20 ?

- Refers 
    - Train and Validate: https://keras.io/examples/audio/transformer_asr/
    - Mel: https://keras.io/examples/audio/melgan_spectrogram_inversion/
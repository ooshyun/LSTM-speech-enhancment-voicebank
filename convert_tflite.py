"""Convert model to TFlite
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

def main():
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))

  # 1. Convert model to float32 (Tf lite)
  model_path = Path(f'result/lstm/20221224-082948')
  saved_model = (model_path / 'model').as_posix()

  # 2. Convert model to TFlite
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model) # load
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS    # enable TensorFlow ops.
  ]

  tflite_float_model = converter.convert() # convert
  with open(model_path / 'model.tflite', 'wb') as f:
      f.write(tflite_float_model)

  # 3. Validate TF Lite Model
  ## 3.1 Load TFLite model
  saved_model_path = (model_path / 'model.tflite').as_posix()

  ## 3.2 Load dataset
  from src.utils import load_yaml
  from src.distrib import load_dataset

  args = load_yaml("./conf/config.yaml")
  train_dataset, test_dataset = load_dataset(args)
  
  tflite_interpreter_float = tf.lite.Interpreter(model_path=saved_model_path)

  ## 3.3 Learn about its input and output details
  input_details = tflite_interpreter_float.get_input_details()
  output_details = tflite_interpreter_float.get_output_details()
  tflite_interpreter_float.allocate_tensors()

  ## 3.4 Evaluate
  for data in test_dataset:
    noisy, clean = data
    
    print(f"Test input shape: {noisy.shape}, {clean.shape}")

    tflite_float_model_predictions = []
    for ibatch in range(noisy.shape[0]):
      one_batch = noisy[np.newaxis, ibatch, ...]
      tflite_interpreter_float.set_tensor(input_details[0]['index'], one_batch)
      tflite_interpreter_float.invoke()
      tflite_float_model_predictions.append(tflite_interpreter_float.get_tensor(output_details[0]['index']))
    
    tflite_float_model_predictions = np.concatenate(tflite_float_model_predictions, axis=0)

    print(f"Test Output shape: {tflite_float_model_predictions.shape}")      

if __name__=="__main__":
  main()
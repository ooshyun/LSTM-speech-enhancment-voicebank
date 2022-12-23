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

  # 2. Convert model float32 -> int8
  ## 2.1 Load representative dataset
  from src.utils import load_yaml
  from src.distrib import load_dataset

  args = load_yaml("./conf/config.yaml")
  train_dataset, test_dataset = load_dataset(args)

  def representative_dataset():
    for data in test_dataset:
      noisy, clean = data
      yield [noisy]

  ## 2.2 Convert model
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model) 
  converter.representative_dataset = representative_dataset
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,    # enable TensorFlow ops.
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

  converter._experimental_lower_tensor_list_ops = False
  # converter.experimental_new_converter = True
  # converter.inference_input_type = tf.int8
  # converter.inference_output_type = tf.int8
  quantized_tflite_model = converter.convert()
  with open(model_path / 'model_int8.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

  # 3. Validate quanitzed model
  ## 3.1 Load quantized TFLite model
  model_path = (model_path / 'model_int8.tflite').as_posix()
  tflite_interpreter_quant = tf.lite.Interpreter(model_path=model_path)
  
  ## 3.2 Learn about its input and output details, and if needs, load scale and zero point
  input_details = tflite_interpreter_quant.get_input_details()
  output_details = tflite_interpreter_quant.get_output_details()
  tflite_interpreter_quant.allocate_tensors()

  # input_scale, input_zero_point = input_details[0]["quantization"]
  # output_scale, output_zero_point = output_details[0]["quantization"]

  ## 3.3 Evaluate depending on the quanitzation method
  for data in test_dataset:
    noisy, clean = data
    
    print(f"Test input shape: {noisy.shape}, {clean.shape}")

    tflite_quantized_model_predictions = []
    for ibatch in range(noisy.shape[0]):
        one_batch = noisy[np.newaxis, ibatch, ...]

        # Validate Quantized int8 model
        # q_one_batch = np.array(one_batch, dtype=np.uint8)
        # q_one_batch = np.array(q_one_batch-128, dtype=np.int8)

        # Validate int8 model using floor and inputscale and zeropoint 
        # one_batch = np.clip(np.floor(one_batch / input_scale + input_zero_point), -128, 127) # for int8 validation
        # q_one_batch = np.array(one_batch, dtype=np.int8)

        tflite_interpreter_quant.set_tensor(input_details[0]['index'], one_batch)
        tflite_interpreter_quant.invoke()
        tflite_quantized_model_predict = tflite_interpreter_quant.get_tensor(output_details[0]['index'])

        # Validate int8 model using floor and inputscale and zeropoint 
        # tflite_quantized_model_predict = (np.array(tflite_quantized_model_predict, dtype=np.float32) - output_zero_point) * output_scale
        
        tflite_quantized_model_predictions.append(tflite_quantized_model_predict)

    tflite_quantized_model_predictions = np.concatenate(tflite_quantized_model_predictions, axis=0)

    print(f"Test Output shape: {tflite_quantized_model_predictions.shape}")      

if __name__=="__main__":
  main()
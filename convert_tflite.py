import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
from src.preprocess.VoiceBankDEMAND import VoiceBandDEMAND
from src.preprocess.feature_extractor import FeatureExtractor
from src.utils import read_audio, load_yaml
from src.distrib import load_model

# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from tensorflow.python.client import device_lib

# SHOULD PUT model path
model_path = Path(f'./history/221122-1127/data/20221123-183326')
path_conf = os.path.join(model_path, "config.yaml")
args = load_yaml(path_conf)

# # 5-1. Generate TF Lite float32 model
# # Convert model to float32 (Tf lite)
saved_model = (model_path / 'model').as_posix()
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model) # load

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter._experimental_lower_tensor_list_ops = False

# MLIR V1 optimization pass is not enabled
# disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.

# Model Compatability https://www.tensorflow.org/lite/guide/ops_select

# 2022-12-16 17:05:31.427644: W tensorflow/compiler/mlir/lite/flatbuffer_export.cc:1901] TFLite interpreter needs to link Flex delegate in order to run the model since it contains the following Select TFop(s):
# Flex ops: FlexTensorListFromTensor, FlexTensorListGetItem, FlexTensorListReserve, FlexTensorListSetItem, FlexTensorListStack
# Details:
#         tf.TensorListFromTensor(tensor<64x?x128xf32>, tensor<2xi32>) -> (tensor<!tf_type.variant<tensor<?x128xf32>>>) : {device = ""}
#         tf.TensorListFromTensor(tensor<?x?x256xf32>, tensor<2xi32>) -> (tensor<!tf_type.variant<tensor<?x256xf32>>>) : {device = ""}
#         tf.TensorListGetItem(tensor<!tf_type.variant<tensor<?x128xf32>>>, tensor<i32>, tensor<2xi32>) -> (tensor<?x128xf32>) : {device = ""}
#         tf.TensorListGetItem(tensor<!tf_type.variant<tensor<?x256xf32>>>, tensor<i32>, tensor<2xi32>) -> (tensor<?x256xf32>) : {device = ""}
#         tf.TensorListReserve(tensor<2xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<?x256xf32>>>) : {device = ""}
#         tf.TensorListSetItem(tensor<!tf_type.variant<tensor<?x256xf32>>>, tensor<i32>, tensor<?x256xf32>) -> (tensor<!tf_type.variant<tensor<?x256xf32>>>) : {device = ""}
#         tf.TensorListStack(tensor<!tf_type.variant<tensor<?x256xf32>>>, tensor<2xi32>) -> (tensor<?x?x256xf32>) : {device = "", num_elements = -1 : i64}
# See instructions: https://www.tensorflow.org/lite/guide/ops_select

tflite_float_model = converter.convert() # convert
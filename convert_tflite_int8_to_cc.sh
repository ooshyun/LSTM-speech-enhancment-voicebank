# Install xxd if it is not available in linux
# xxd alread exists in macos
# sudo apt-get update && apt-get -qq install xxd 

# Convert to a C source file
# sudo xxd -i ./keras_lstm/model_quantized_minispeech.tflite > ./keras_lstm/model.cc

MODEL_PATH="./result/lstm/20221224-082948"
MODEL_NAME="/model_int8"
_MODEL_PATH_TFLITE="$MODEL_PATH$MODEL_NAME.tflite"
_MODEL_PATH_CC="$MODEL_PATH$MODEL_NAME.cc"

sudo xxd -i $_MODEL_PATH_TFLITE > $_MODEL_PATH_CC

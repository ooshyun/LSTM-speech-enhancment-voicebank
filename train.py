import os
import datetime
import numpy as np
from pathlib import Path
from shutil import copyfile
# import IPython.display as ipd

# Tensor related library
# Load the TensorBoard notebook extension.
# %load_ext tensorboard
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras.models
import keras.callbacks

# custom api
from src.distrib import load_dataset
from src.utils import TimeHistory, load_yaml, save_json

def train(args):
    # 1. Set Paramter
    device_lib.list_local_devices()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.name

    train_dataset, test_dataset = load_dataset(args)

    # 3. Build model
    if model_name == "cnn":
        from src.cnn import build_model
        model = build_model(l2_strength=0.0, args=args)
    elif model_name == "lstm":
        from src.lstm import build_model_lstm
        model = build_model_lstm(args)
    else:
        raise ValueError("Model didn't implement...")
    model.summary()

    if args.model.path is not None:
        if args.model.ckpt:
            model.load_weights(os.path.join(args.model.path, args.model.ckpt))
        else:
            model = keras.models.load_model(os.path.join(args.model.path, "model"), compile=False)

    if model_name == "cnn":
        from src.cnn import compile_model
        compile_model(model, args)

    elif model_name == "lstm":
        from src.lstm import compile_model
        compile_model(model, args)    
            
    # You might need to install the following dependencies: sudo apt install python-pydot python-pydot-ng graphviz
    # keras.utils.plot_model(model, show_shapes=True, dpi=64)

    # 4. Set logging
    save_path = os.path.join(args.folder, model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    print("Save path: ", save_path)

    checkpoint_save_path = os.path.join(save_path, "checkpoint/checkpoint-{epoch:02d}-{val_loss:.9f}.hdf5")
    model_save_path = os.path.join(save_path, "model")
    optimizer_save_path = os.path.join(save_path, "optimizer")
    console_log_save_path = os.path.join(save_path, "debug.txt")

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, baseline=None)
    logdir = os.path.join(f"./logs/{model_name}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, update_freq='batch', histogram_freq=1, write_graph=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, 
                                                            test='val_loss', save_best_only=True)
    time_callback = TimeHistory(filepath=console_log_save_path)
    # histogram_freq=0, write_graph=True: for monitoring the weight histogram
    
    # 5. Evaluate model
    baseline_val_loss = model.evaluate(test_dataset)[0]
    print(f"Baseline accuracy {baseline_val_loss}")

    # 6. Train
    model.fit(train_dataset, # model.fit([pair_1, pair_2], labels, epochs=50)
            steps_per_epoch=args.steps, # you might need to change this
            validation_data=test_dataset,
            epochs=args.epochs,
            callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback, time_callback]
            )

    # 7. Save trained model after evaluation
    val_loss = model.evaluate(test_dataset)[0]
    if val_loss < baseline_val_loss:
        print("New model saved.")
        keras.models.save_model(model, model_save_path, overwrite=True, include_optimizer=True)
        optimizer_save_path = Path(optimizer_save_path)
        if not optimizer_save_path.is_dir():
            optimizer_save_path.mkdir()
        optimizer_save_path = optimizer_save_path / "optim.json"
        save_json({"optimizer":model.optimizer.get_weights()}, optimizer_save_path)
        
    return save_path


if __name__=="__main__":
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    path_conf = "./conf/config.yaml"
    config = load_yaml(path_conf)
    save_path = train(config)

    copyfile(path_conf, f"{save_path}/config.yaml")
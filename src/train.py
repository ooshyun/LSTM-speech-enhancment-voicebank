import os
import datetime
import numpy as np
from shutil import copyfile

# import IPython.display as ipd

# Tensor related library
# Load the TensorBoard notebook extension.
# %load_ext tensorboard
import tensorflow as tf
from tensorflow.python.client import device_lib

# custom api
from src.distrib import load_dataset, load_model, load_callback
from src.utils import load_yaml, obj2dict, limit_gpu_tf


def train(path_conf):
    # 1. Set Paramter
    device_lib.list_local_devices()
    args = load_yaml(path_conf)

    print("  Train Parameter")
    args_dict = obj2dict(args)
    print(args_dict)
    print("------")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.name

    train_dataset, test_dataset = load_dataset(args)

    # 3. Build model
    model = load_model(args)

    # You might need to install the following dependencies: sudo apt install python-pydot python-pydot-ng graphviz
    # keras.utils.plot_model(model, show_shapes=True, dpi=64)

    # 4. Set logging
    save_path = os.path.join(
        args.folder, model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    copyfile(path_conf, f"{save_path}/config.yaml")
    
    callbacks_list = load_callback(save_path, args)
    print("Save path: ", save_path)

    # 5. Evaluate model
    baseline_val_loss = model.evaluate(test_dataset)[0]
    print(f"Baseline accuracy {baseline_val_loss}")

    # 6. Train
    model.fit(
        train_dataset,  # model.fit([pair_1, pair_2], labels, epochs=50)
        steps_per_epoch=args.steps,  # you might need to change this
        validation_data=test_dataset,
        epochs=args.epochs,
        callbacks=callbacks_list,
    )

    # 7. Save trained model after evaluation
    val_loss = model.evaluate(test_dataset)[0]
    
    if val_loss < baseline_val_loss:
        print("New model saved.")
        from src.distrib import save_model_all

        save_model_all(save_path, model)

def main(gpu_size, path_conf):
    limit_gpu_tf(gpu_size) # 12G
    train(path_conf)

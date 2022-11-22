from keras.layers import (
    Conv2D,
    Input,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    SpatialDropout2D,
)
import os
from .utils import load_json
from keras import Model
import keras.regularizers
import keras.optimizers

import tensorflow as tf


def conv_block(x, filters, kernel_size, strides, padding="same", use_bn=True):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(0.0006),
    )(x)
    x = Activation("relu")(x)
    if use_bn:
        x = BatchNormalization()(x)
    return x


def full_pre_activation_block(
    x, filters, kernel_size, strides, padding="same", use_bn=True
):
    shortcut = x
    in_channels = x.shape[-1]

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding="same"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters=in_channels, kernel_size=kernel_size, strides=strides, padding="same"
    )(x)

    return shortcut + x


def build_model(l2_strength, args):
    inputs = Input(shape=[args.dset.n_feature, args.dset.n_segment, 1])
    x = inputs

    # -----
    x = ZeroPadding2D(((4, 4), (0, 0)))(x)
    x = Conv2D(
        filters=18,
        kernel_size=[9, 8],
        strides=[1, 1],
        padding="valid",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    skip0 = Conv2D(
        filters=30,
        kernel_size=[5, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(skip0)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=8,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # -----
    x = Conv2D(
        filters=18,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    skip1 = Conv2D(
        filters=30,
        kernel_size=[5, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(skip1)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=8,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(
        filters=18,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=30,
        kernel_size=[5, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=8,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(
        filters=18,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=30,
        kernel_size=[5, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = x + skip1
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=8,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(
        filters=18,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=30,
        kernel_size=[5, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = x + skip0
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        filters=8,
        kernel_size=[9, 1],
        strides=[1, 1],
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # ----
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding="same")(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def compile_model(model: Model, args):
    optimizer = keras.optimizers.Adam(3e-4)
    # optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=3e-4)

    if args.model.path is not None:
        tf.print("Optimizer Loading...")
        dummpy_model = build_model(args)
        optimizer_state = load_json(
            os.path.join(args.model.path, "optimizer/optim.json")
        )["optimizer"]
        dummy_batch_size = 1
        dummy_noise_tensor = tf.ones(
            shape=(dummy_batch_size, args.dset.n_feature, args.dset.n_segment, 1)
        )
        dummy_clean_tensor = tf.ones(
            shape=(dummy_batch_size, args.dset.n_feature, 1, 1)
        )
        dummpy_model.compile(
            optimizer=optimizer,
            loss="mse",
        )
        dummpy_model.fit(
            x=dummy_noise_tensor, y=dummy_clean_tensor, batch_size=dummy_batch_size
        )
        del (
            dummpy_model,
            dummpy_model,
            dummy_noise_tensor,
            dummy_clean_tensor,
        )  # [TODO] How to remove object and check it removed?
        optimizer.set_weights(optimizer_state)
        tf.print("Optimizer was loaded!")

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError("rmse")],
    )



import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2


def build_model(frames=128, bands=128, channels=1, num_labels=10,
                kernel=(5,5), pool=(4,2),
                fully_connected=64,
                kernels_start=24, kernels_growth=2,
                dropout=0.5,
                use_strides=True,
                depthwise_separable=True):
    """
    Implements SB-CNN model from
    Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
    Salamon and Bello, 2016.
    https://arxiv.org/pdf/1608.04363.pdf

    Based on https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae
    but parameters are changed back to those of the original paper authors,
    and added Batch Normalization
    """
    Conv2 = SeparableConv2D if depthwise_separable else Convolution2D
    if use_strides:
        strides = pool
        pool = (1, 1)
    else:
        strides = (1, 1)

    block1 = [
        Convolution2D(kernels_start, kernel, padding='same', strides=strides,
                      input_shape=(bands, frames, channels)),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool),
        Activation('relu'),
    ]
    block2 = [
        Conv2(kernels_start*kernels_growth, kernel, padding='same', strides=strides),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool),
        Activation('relu'),
    ]
    block3 = [
        Conv2(kernels_start*kernels_growth, kernel, padding='valid', strides=strides),
        BatchNormalization(),
        Activation('relu'),
    ]
    backend = [
        Flatten(),

        Dense(fully_connected, kernel_regularizer=l2(0.001)),
        Activation('relu'),
        Dropout(dropout),

        Dense(num_labels, kernel_regularizer=l2(0.001)),
        Dropout(dropout),
        Activation('softmax'),
    ]
    layers = block1 + block2 + block3 + backend
    model = Sequential(layers)
    return model



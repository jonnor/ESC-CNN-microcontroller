

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2


def build_model(frames=128, bands=128, channels=1, num_labels=10,
                kernel=(5,5), pool=(4,2), dropout=0.5, depthwise_separable=False):
    """
    Implements SB-CNN model from
    Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
    Salamon and Bello, 2016.
    https://arxiv.org/pdf/1608.04363.pdf

    Based on https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae
    but parameters are changed back to those of the original paper authors
    """

    model = Sequential([])

    Conv2 = SeparableConv2D if depthwise_separable else Convolution2D

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the shape (24,1,f,f). 
    # This is followed by (4,2) max-pooling over the last two dimensions and a ReLU activation function.
    model.add(Convolution2D(24, kernel, padding='same', input_shape=(bands, frames, channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool))
    model.add(Activation('relu'))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48,24,f,f). 
    # Like L1 this is followed by (4,2) max-pooling and a ReLU activation function.
    model.add(Conv2(48, kernel, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool))
    model.add(Activation('relu'))

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48, 48, f, f). 
    # This is followed by a ReLU but no pooling.
    model.add(Conv2(48, kernel, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # flatten output into a single dimension, let Keras do shape inference
    model.add(Flatten())

    # Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # Layer 5 - an output layer with one output unit per class, with L2 penalty, 
    # followed by a softmax activation function
    model.add(Dense(num_labels, kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout))
    model.add(Activation('softmax'))

    return model



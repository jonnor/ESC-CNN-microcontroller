

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2


def build_model(frames=41, bands=60, num_channels=1, num_labels=10, f_size=3):
    """
    Implements SB-CNN model from
    Deep Convolutional Neural Networks and DataAugmentation for Environmental SoundClassification
    Salamon and Bello, 2016.
    https://arxiv.org/pdf/1608.04363.pdf

    Based on https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae
    but parameters are changed back to those of the original paper authors
    """

    model = Sequential()

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the shape (24,1,f,f). 
    # This is followed by (4,2) max-pooling over the last two dimensions and a ReLU activation function.
    model.add(Convolution2D(24, f_size, f_size, border_mode='same', input_shape=(bands, frames, num_channels)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48,24,f,f). 
    # Like L1 this is followed by (4,2) max-pooling and a ReLU activation function.
    model.add(Convolution2D(48, f_size, f_size, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48, 48, f, f). 
    # This is followed by a ReLU but no pooling.
    model.add(Convolution2D(48, f_size, f_size, border_mode='valid'))
    model.add(Activation('relu'))

    # flatten output into a single dimension, let Keras do shape inference
    model.add(Flatten())

    # Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
    model.add(Dense(64, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 5 - an output layer with one output unit per class, with L2 penalty, 
    # followed by a softmax activation function
    model.add(Dense(num_labels, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    return model

def main():
    m = build_model()

    m.summary()

if __name__ == '__main__':
    main()




import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D
from keras.regularizers import l2


def build_model(frames=128, bands=128, channels=1, num_labels=10, kernel=(5,5), pool=(4,2), dropout=0.5, depthwise_separable=False):
    """
    Implements SB-CNN model from
    Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
    Salamon and Bello, 2016.
    https://arxiv.org/pdf/1608.04363.pdf

    Based on https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae
    but parameters are changed back to those of the original paper authors
    """

    model = Sequential()

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
    model.add(Conv2(num_labels, kernel, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    return model

def main():
    print('sbcnn.orig')
    m = build_model()
    m.summary()
    m.save('sbcnn.orig.hdf5')

    print('sbcnn16k30')
    m = build_model(frames=72, bands=30, kernel=(3,3), pool=(3,3))
    m.summary()
    m.save('sbcnn16k30.hdf5')



if __name__ == '__main__':
    main()


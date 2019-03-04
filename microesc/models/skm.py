


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2


def build_model(frames=172, shingles=8, bands=40, channels=1, codebook=2000):
    """
    Implements convolution part of SKM model from

    UNSUPERVISED FEATURE LEARNING FOR URBAN SOUND CLASSIFICATION
    Justin Salamon and Juan Pablo Bello, 2015
    """

    input_shape=(bands, frames, channels)
    kernel = (bands, shingles)

    model = Sequential([
        Convolution2D(codebook, kernel, strides=(1,shingles) , padding='same', activation=None, input_shape=input_shape)
    ])

    return model

def main():
    print('original')
    m = build_model()
    m.summary()



if __name__ == '__main__':
    main()


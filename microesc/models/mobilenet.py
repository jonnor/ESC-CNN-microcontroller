

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2


def build_model(frames=32, bands=32, channels=1, num_labels=10,
                pretrain=None,
                alpha=0.35, depth_multiplier=0.5):
    """
    """

    from keras.applications import mobilenet_v2

    input_shape = (frames, bands, channels)
    model = mobilenet_v2.MobileNetV2(weights=pretrain, classes=1000,
                    input_shape=input_shape, include_top=True,
                    depth_multiplier=depth_multiplier,
                    alpha=alpha)


    return model

def main():
    m = build_model()

    m.summary()

if __name__ == '__main__':
    main()


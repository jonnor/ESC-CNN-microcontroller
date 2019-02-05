
import keras

def build_tiny_conv(input_frames, input_bins, n_classes=12, dropout=0.5):
    """
    Ported from Tensorflow examples. create_tiny_conv_model
    """

    from keras.layers import Conv2D, Dense, Dropout, Flatten
    input_shape = (input_bins, input_frames, 1)

    model = keras.Sequential([
        Conv2D(8, (8, 10), strides=(2, 2),
                padding='same', activation='relu', use_bias=True,
                input_shape=input_shape),
        Dropout(dropout),
        Flatten(),
        Dense(n_classes, activation='softmax', use_bias=True),
    ])
    return model

def build_low_latency_conv(input_frames, input_bins, n_classes=12, dropout=0.5):
    """
    Ported from Tensorflow examples. create_low_latency_conv

    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    """

    from keras.layers import Conv2D, Dense, Dropout, Flatten
    input_shape = (input_frames, input_bins, 1)

    # In the paper there are some differences
    # uses log-mel as input instead of MFCC
    # uses 4 in stride for frequency
    # has a linear bottleneck as second layer to reduce multiplications,
    # instead of doing a single full-frequency convolution
    # probably uses ReLu for the DNN layers?
    # probably does not use ReLu for the conv layer?

    # Note, in keyword spotting task tstride=2,4,8 performed well also
    model = keras.Sequential([
        Conv2D(186, (input_frames, 8), strides=(1, 1),
                padding='valid', activation='relu', use_bias=True,
                input_shape=input_shape),
        Dropout(dropout),
        Flatten(),
        Dense(128, activation=None, use_bias=True),
        Dropout(dropout),
        Dense(128, activation=None, use_bias=True),
        Dropout(dropout),
        Dense(n_classes, activation='softmax', use_bias=True),
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_aclnet_lowlevel(input_samples, c1=32, s1=8, s2=4, input_tensor=None):

    """

    The following values were tested in the paper.
    c1= 8,16,32
    s1= 2,4,8
    s2= 2,4
    """    

    from keras.layers import Conv1D, MaxPooling1D, InputLayer, Flatten, Dense
    input_shape = (input_samples, 1)

    model = keras.Sequential([
        InputLayer(input_shape=input_shape, input_tensor=input_tensor),
        Conv1D(filters=c1, kernel_size=9, strides=s1,
                padding='valid', activation=None, use_bias=False,),
        Conv1D(filters=64, kernel_size=5, strides=s2,
                padding='valid', activation=None, use_bias=False,),
        MaxPooling1D(pool_size=(int(160/(s2*s1)),),
                    padding='valid', data_format='channels_last'),
        Flatten(),
        Dense(1, activation=None),
    ])

    return model


def main():
    m = build_low_latency_conv(98, 40)
    m.summary()

    m = build_tiny_conv(32, 40)
    m.summary()

    m = build_aclnet_lowlevel(20480)
    m.summary()

if __name__ == '__main__':
    main()


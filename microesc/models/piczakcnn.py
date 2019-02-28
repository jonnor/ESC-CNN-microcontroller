

# NOTE: log-melspectrogram and delta log-melspec
def build_model(bands=60, frames=41, channels=2, n_labels=10,
                fc=5000, dropout=0.5):

    """
    Implements the short-segment CNN from

    ENVIRONMENTAL SOUND CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORKS
    Karol J. Piczak, 2015.
    https://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
    """

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.regularizers import l2

    input_shape = (bands, frames, channels)

    model = Sequential([
        Convolution2D(80, (bands-3,6), strides=(1,1), input_shape=input_shape),
        MaxPooling2D((4,3), strides=(1,3)),
        Convolution2D(80, (1,3)),
        MaxPooling2D((1,3), strides=(1,3)),
        #Flatten(),
        Dense(fc, activation='relu'),
        Dropout(dropout),
        Dense(fc, activation='relu'),
        Dropout(dropout),
        Dense(n_labels, activation='softmax'),
    ])

    return model


def main():
    m = build_model()

    m.summary()

if __name__ == '__main__':
    main()

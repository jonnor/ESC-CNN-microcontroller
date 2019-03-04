

def build_model(bands=128, frames=128, channels=2, n_classes=10,
            filters=80, L=57, W=6, fully_connected=5000):

    """
    Deep Convolutional Neural Network with Mixup for Environmental Sound Classification
   
    https://link.springer.com/chapter/10.1007/978-3-030-03335-4_31
    """

    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, Input, Concatenate
    from keras.layers import Convolution2D, Flatten, MaxPooling2D
    import keras.layers
    
    input_shape = (bands, frames, channels)
   
    # FIXME: add missing BatchNormalization and Dropout and L2 regularization
    model = Sequential([
        Convolution2D(32, (3,7), padding='same', input_shape=input_shape),
        Convolution2D(32, (3,5), padding='same'),
        MaxPooling2D(pool_size=(4,3)),
        Convolution2D(64, (3,1), padding='same'),
        Convolution2D(64, (3,1), padding='same'),
        MaxPooling2D(pool_size=(4,1)),
        Convolution2D(128, (1,5), padding='same'),
        Convolution2D(128, (1,5), padding='same'),
        MaxPooling2D(pool_size=(1,3)),
        Convolution2D(256, (3,3), padding='same'),
        Convolution2D(256, (3,3), padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Dense(512, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    return model


def main():
    m = build_model()
    m.summary()
    m.save('dmix.orig.hdf5')


if __name__ == '__main__':
    main()

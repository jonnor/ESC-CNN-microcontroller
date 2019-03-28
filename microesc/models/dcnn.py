
def dcnn_head(input, head_name, filters=80, kernel=(3,3)):
    def n(base):
        return base+'_'+head_name
    
    from keras.layers import Convolution2D, Flatten, MaxPooling2D

    x = input
    x = Convolution2D(filters, kernel, dilation_rate=(1,1), name=n('DilaConv1'))(x)
    x = MaxPooling2D(pool_size=(4,3), name=n('MPL1'))(x)
    x = Convolution2D(filters, kernel, dilation_rate=(2,2), name=n('DilaConv2'))(x)
    x = MaxPooling2D(pool_size=(1,3), name=n('MPL2'))(x)

    x = Flatten(name=n('flatten'))(x)
    return x

def dcnn(bands=60, frames=31, n_classes=10, fully_connected=5000, filters=80, activation='relu'):
    """
    Dilated Convolution Neural Network with LeakyReLU for Environmental Sound Classification

    https://ieeexplore.ieee.org/document/8096153
    """
    # XXX: kernel size is missing from paper    

    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, Input, Concatenate
    import keras.layers

    input_shape = (bands, frames, 1)

    def head(input, name):
        return dcnn_head(input, name, filters)

    mel_input = Input(shape=input_shape, name='mel_input')
    delta_input = Input(shape=input_shape, name='delta_input')
    heads = [
        head(mel_input, 'mel'),
        head(delta_input, 'delta')
    ]
    m = keras.layers.concatenate(heads)
    m = Dense(fully_connected, activation=activation)(m)
    m = Dense(fully_connected, activation=activation)(m)
    m = Dense(n_classes, activation='softmax')(m)

    model = Model([mel_input, delta_input], m)

    return model


def dcnn_nodelta(bands=60, frames=31, n_classes=10, channels=1,
                 fully_connected=5000, filters=80, activation='relu'):

    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, Input, Concatenate
    import keras.layers

    input_shape = (bands, frames, channels)
    def head(input, name):
        return dcnn_head(input, name, filters)

    mel_input = Input(shape=input_shape, name='mel_input')
    m = head(mel_input, 'mel')
    m = Dense(fully_connected, activation=activation)(m)
    m = Dense(fully_connected, activation=activation)(m)
    m = Dense(n_classes, activation='softmax')(m)

    model = Model(mel_input, m)
    return model


def main():
    m = dcnn()
    m.save('dcnn.hdf5')
    m.summary()

    m = dcnn_nodelta()
    m.save('dcnn.nodelta.hdf5')
    m.summary()

if __name__ == '__main__':
    main()

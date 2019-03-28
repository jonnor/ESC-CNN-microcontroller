
import keras

def ldcnn_head(input, head_name, filters=80, L=57, W=6):
    def n(base):
        return base+'_'+head_name
    
    from keras.layers import Convolution2D, Flatten, MaxPooling2D, BatchNormalization

    x = input
    x = Convolution2D(filters, (L,1), activation='relu', name=n('SFCL1'))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(filters, (1,W), activation='relu', name=n('SFCL2'))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4,3), strides=(1,3), name=n('MPL1'))(x)
    x = Convolution2D(filters, (1,3), dilation_rate=(2,2), name=n('DCL'))(x)
    x = MaxPooling2D(pool_size=(1,3), strides=(1,3), name=n('MPL2'))(x)
    x = Flatten(name=n('flatten'))(x)
    return x


def ldcnn(bands=60, frames=31, n_classes=10,
            filters=80, L=57, W=6, fully_connected=5000, dropout=0.25):

    """
    LD-CNN: A Lightweight Dilated Convolutional Neural Network for Environmental Sound Classification
    
    http://epubs.surrey.ac.uk/849351/1/LD-CNN.pdf
    """

    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, Input, Concatenate
    from keras.regularizers import l2
    import keras.layers

    input_shape = (bands, frames, 1)

    def head(input, name):
        return ldcnn_head(input, name, filters, L, W)

    mel_input = Input(shape=input_shape, name='mel_input')
    delta_input = Input(shape=input_shape, name='delta_input')
    heads = [
        head(mel_input, 'mel'),
        head(delta_input, 'delta')
    ]
    m = keras.layers.add(heads, name='FSL')
    m = Dropout(dropout)(m)
    m = Dense(fully_connected, activation='relu', kernel_regularizer=l2(0.001), name='FCL')(m)
    m = Dropout(dropout)(m)
    m = Dense(n_classes, activation='softmax')(m)

    model = Model([mel_input, delta_input], m)

    return model



def ldcnn_nodelta(bands=60, frames=31, n_classes=10,
            filters=80, L=57, W=6, channels=1, fully_connected=5000, dropout=0.5):
    """Variation of LD-CNN with only mel input (no deltas)"""

    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, Input, Concatenate
    from keras.regularizers import l2

    input_shape = (bands, frames, channels)
    input = Input(shape=input_shape, name='mel_input')

    m = ldcnn_head(input, 'mel', filters, L, W)
    #m = Dropout(dropout)(m)

    m = Dense(fully_connected, activation='relu', kernel_regularizer=l2(0.001), name='FCL')(m)
    m = Dropout(dropout)(m)

    m = Dense(n_classes, kernel_regularizer=l2(0.001))(m)
    m = Dropout(dropout)(m)
    m = Activation('softmax')(m)

    model = Model(input, m)
    return model


def main():
    m = ldcnn()
    m.save('ldcnn.hdf5')
    m.summary()

    m = ldcnn_nodelta()
    m.save('ldcnn.nodelta.hdf5')
    m.summary()

if __name__ == '__main__':
    main()

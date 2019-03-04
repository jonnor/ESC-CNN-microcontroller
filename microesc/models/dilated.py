
def build_model(bands=64, frames=41, channels=2,
                dilation=(2,2), kernel=(3,3), n_labels=10, dropout=0.5,
                kernels=[32, 32, 64, 64]):
    """
    Environmental sound classification with dilated convolutions

    https://www.sciencedirect.com/science/article/pii/S0003682X18306121
    """

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, GlobalAveragePooling2D
    from keras.regularizers import l2
    
    input_shape = (bands, frames, channels)

    # XXX: number of kernels in original paper is unknown
    conv = [
        Convolution2D(kernels[0], kernel, input_shape=input_shape, activation='relu')
    ]
    for k in kernels[1:]:
        c = Convolution2D(k, kernel, dilation_rate=dilation, activation='relu')
        conv.append(c)

    model = Sequential(conv + [
        GlobalAveragePooling2D(),
        Dropout(dropout),
        Dense(n_labels, activation='softmax'),
    ])

    return model

def main():
    m = build_model()

    m.summary()

if __name__ == '__main__':
    main()

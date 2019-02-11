
def build_model(bands=60, frames=41, channels=2,
                n_labels=10, dropout=0.0):
    """
    Environmental sound classification with dilated convolutions

    https://www.sciencedirect.com/science/article/pii/S0003682X18306121
    """

    from keras_contrib.applications import densenet
    
    input_shape = (bands, frames, channels)

    model = densenet.DenseNet(input_shape=input_shape,
                                depth=10, nb_dense_block=3, growth_rate=12,
                                classes=n_labels, dropout_rate=dropout)

    return model

def main():
    m = build_model()

    m.summary()

if __name__ == '__main__':
    main()

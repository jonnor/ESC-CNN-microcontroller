

"""

Inspired by CRNN model described in

Sound Event Detection: A Tutorial
https://arxiv.org/abs/2107.05463

and

Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection
https://arxiv.org/abs/1702.06286
"""

# related code, https://chadrick-kwag.net/tf-keras-rnn-ctc-example/

def build_model(frames=128, bands=40, channels=1, n_classes=10,
                conv_size=(3,3),
                conv_block='conv',
                downsample_size=(2,2),
                n_stages=3, n_blocks_per_stage=1,
                filters=128, kernels_growth=1.0,
                fully_connected=64,
                rnn_units=32,
                temporal='bigru',
                dropout=0.5, l2=0.001, backend='detection'):


    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.layers import \
        Conv2D, LSTM, GRU, Bidirectional, MaxPooling2D, \
        Reshape, TimeDistributed, Softmax, Dense, SeparableConv2D

    model = Sequential()

    input_shape = (frames, bands, channels)

    def add_conv_block(model, downsample_size, conv_filters=filters, kernel_size=conv_size,
                    **kwargs):
        model.add(SeparableConv2D(conv_filters, conv_size, **kwargs))
        model.add(MaxPooling2D(downsample_size))

        # TODO: add ReLu
        # TODO: BatchNorm etc?

    # Convolutional layers
    add_conv_block(model, downsample_size=(1,5), input_shape=input_shape)
    add_conv_block(model, downsample_size=(1,2))
    add_conv_block(model, downsample_size=(1,2))

    # Temporal processing
    if temporal == 'bigru':
        o = model.layers[-1].output_shape
        model.add(Reshape((o[1], -1)))
        model.add(Bidirectional(GRU(rnn_units, return_sequences=True)))
        model.add(Bidirectional(GRU(rnn_units, return_sequences=True)))
    elif temporal == 'tcn':
        # TODO: make downsampling adjustable
        model.add(SeparableConv2D(rnn_units, (9, 1), strides=(2,1)))
        model.add(SeparableConv2D(rnn_units, (9, 1), strides=(2,1)))
    else:
        raise ValueError(f"Unknown temporal parameter {temporal}")

    # Output
    # TODO: support multiple layers
    # TODO: add Dropout
    o = model.layers[-1].output_shape   
    if backend == 'classification':
        model.add(TimeDistributed(Dense(fully_connected, activation="linear")))
        model.add(layers.Dense(n_classes))
        model.add(Softmax())

    elif backend == 'detection':
        #model.add(TimeDistributed(Dense(fully_connected, activation="linear")))
        model.add(TimeDistributed(Dense(n_classes, activation="linear"), input_shape=(o[1], o[2])))
        model.add(Softmax())
    elif not backend:
        pass # no backend
    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return model


def test_model():

    model = build_model(filters=24, bands=64, rnn_units=16, n_classes=3, temporal='tcn')

    print(model.summary())


if __name__ == '__main__':
    test_model()
    

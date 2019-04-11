

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import MaxPooling2D, SeparableConv2D, Conv2D, DepthwiseConv2D


def add_common(x, name):
    x = BatchNormalization(name=name+'_bn')(x)
    x = Activation('relu', name=name+'_relu')(x)
    return x

def conv(x, kernel, filters, downsample, name,
        padding='same'):
    """Regular convolutional block"""
    x = Conv2D(filters, kernel, strides=downsample,
            name=name, padding=padding)(x)
    return add_common(x, name)

def conv_ds(x, kernel, filters, downsample, name,
        padding='same'):
    """Depthwise Separable convolutional block
    (Depthwise->Pointwise)

    MobileNet style"""
    x = SeparableConv2D(filters, kernel, padding=padding, 
                strides=downsample, name=name+'_ds')(x)
    return add_common(x, name=name+'_ds')

def conv_bottleneck_ds(x, kernel, filters, downsample, name,
        padding='same', bottleneck=0.5):
    """
    Bottleneck -> Depthwise Separable
    (Pointwise->Depthwise->Pointswise)

    MobileNetV2 style
    """
    x = Conv2D(int(filters*bottleneck), (1,1),
                padding='same', strides=downsample,
                name=name+'_pw')(x)
    add_common(x, name+'_pw')
    x = SeparableConv2D(filters, kernel,
                padding=padding, strides=(1,1),
                name=name+'_ds')(x)
    return add_common(x, name+'_ds')


def conv_effnet(x, kernel, filters, downsample, name,
        bottleneck=0.5, strides=(1,1), padding='same', bias=False):
    """Pointwise -> Spatially Separable conv&pooling  
    Effnet style"""

    assert downsample[0] == downsample[1]
    downsample = downsample[0]
    assert kernel[0] == kernel[1]
    kernel = kernel[0]

    ch_in = int(filters*bottleneck)
    ch_out = filters

    x = Conv2D(ch_in, (1, 1), strides=downsample,
            padding=padding, use_bias=bias, name=name+'pw')(x)
    x = add_common(x, name=name+'pw')

    x = DepthwiseConv2D((1, kernel),
            padding=padding, use_bias=bias, name=name+'dwv')(x)
    x = add_common(x, name=name+'dwv')

    x = DepthwiseConv2D((kernel, 1), padding='same',
            use_bias=bias, name=name+'dwh')(x)
    x = add_common(x, name=name+'dwh')

    x = Conv2D(ch_out, (1, 1), padding=padding, use_bias=bias, name=name+'rh')(x)
    return add_common(x, name=name+'rh')


block_types = {
    'conv': conv,
    'depthwise_separable': conv_ds,
    'bottleneck_ds': conv_bottleneck_ds,
    'effnet': conv_effnet,
}

def backend_dense1(x, n_classes, fc=64, regularization=0.001, dropout=0.5):
    from keras.regularizers import l2
    """
    SB-CNN style classification backend
    """

    x = Flatten()(x)
    x = Dense(fc, kernel_regularizer=l2(regularization))(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(n_classes, kernel_regularizer=l2(regularization))(x)
    x = Dropout(dropout)(x)
    x = Activation('softmax')(x)
    return x


def build_model(frames=128, bands=128, channels=1, n_classes=10,
                conv_size=(5,5),
                conv_block='conv',
                downsample_size=(2,2),
                n_stages=3, n_blocks_per_stage=1,
                filters=24, kernels_growth=1.5,
                fully_connected=64,
                dropout=0.5, l2=0.001):
    """
    
    """

    conv_func = block_types.get(conv_block)
    input = Input(shape=(bands, frames, channels))
    x = input

    block_no = 0
    for stage_no in range(0, n_stages):
        for b_no in range(0, n_blocks_per_stage):
            padding = 'valid' if block_no == (n_stages*n_blocks_per_stage)-1 else 'same'
            name = "conv{}".format(block_no)
            downsample = downsample_size if b_no == 0 else (1, 1) 
            x = conv_func(x, conv_size, int(filters), downsample,
                            name=name, padding=padding)
            block_no += 1
        filters = filters * kernels_growth

    x = backend_dense1(x, n_classes, fully_connected, regularization=l2)  
    model = Model(input, x)
    return model



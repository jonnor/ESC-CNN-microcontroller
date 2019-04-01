

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2
import keras.layers

from keras import backend, layers

def relu6(x, name):
    if False:
        x = layers.ReLU(6., name=name)(x)
    else:
        x = layers.Activation('relu')(x)
    return x


# From keras_applications.mobilenet
# Modified to work with STM32 Cube.AI 3.3
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    padding = ((0, kernel[1]//2), (0, kernel[1]//2))
    x = layers.ZeroPadding2D(padding=padding, name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return relu6(x, name='conv1_relu')

# From keras_applications.mobilenet
# Modified to work with STM32 Cube.AI 3.3
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), kernel=(3,3), block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    layers = keras.layers

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, kernel[1]//2), (0, kernel[1]//2)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D(kernel,
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = relu6(x, name='conv_dw_%d_relu' % block_id)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return relu6(x, name='conv_pw_%d_relu' % block_id)


def build_model(frames=32, bands=32, channels=1, n_classes=10, dropout=0.5,
                depth_multiplier=1,
                alpha=1.0, n_stages=3,
                initial_filters = 32,
                kernel = (5,5),
                stride_f = 2, stride_t = 2):
    """
    """

    from keras.applications import mobilenet

    conv = _conv_block
    dwconv = _depthwise_conv_block

    assert keras.backend.image_data_format() == 'channels_last'

    input_shape = (bands, frames, channels)
    img_input = keras.layers.Input(shape=input_shape)
    
    x = conv(img_input, initial_filters, alpha, kernel=kernel, strides=(1, 1))
    x = dwconv(x, initial_filters*2, alpha, depth_multiplier, block_id=1)

    for stage_no in range(1, n_stages):
        filters = initial_filters*2**stage_no
        x = dwconv(x, filters, alpha, depth_multiplier,
                    kernel=kernel, strides=(stride_f, stride_t), block_id=(stage_no*2))
        x = dwconv(x, filters, alpha, depth_multiplier,
                    kernel=kernel, block_id=(stage_no*2)+1)

    shape = (1, 1, int(filters * alpha))

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape(shape, name='reshape_1')(x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    x = keras.layers.Conv2D(n_classes, (1, 1), padding='same', name='conv_preds')(x)
    x = keras.layers.Activation('softmax', name='act_softmax')(x)
    x = keras.layers.Reshape((n_classes,), name='reshape_2')(x)

    model = keras.Model(img_input, x)

    return model



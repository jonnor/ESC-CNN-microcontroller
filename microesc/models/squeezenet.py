
"""
SqueezeNet v1.1 implementation
from "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
https://arxiv.org/abs/1602.07360

Based on
https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
"""

import keras
from keras.layers import Convolution2D, Activation


def fire_module(x, fire_id, squeeze=16, expand=64):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    s_id = 'fire' + str(fire_id) + '/'
 
    from keras.layers import concatenate

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)
    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    assert keras.backend.image_data_format() == 'channels_last'
    x = concatenate([left, right], axis=3, name=s_id + 'concat')
    return x

def build_model(frames=32, bands=32, channels=1, n_classes=10,
                dropout=0.5,
                n_stages=3,
                modules_per_stage=2,
                initial_filters=64,
                squeeze_ratio=0.2,
                pool = (2, 2),
                kernel = (3, 3),
                stride_f = 2, stride_t = 2):

    from keras.models import Model
    from keras.layers import Input, GlobalAveragePooling2D, Dropout, MaxPooling2D

    input_shape = (bands, frames, channels)
    img_input = keras.layers.Input(shape=input_shape)
    x = Convolution2D(initial_filters, (3, 3), strides=(stride_f, stride_t),
                      padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)

    module_idx = 0
    for stage_no in range(1, n_stages):
        expand = initial_filters*stage_no
        squeeze = int(expand * squeeze_ratio)
        x = MaxPooling2D(pool_size=pool, strides=(stride_f, stride_t), name='pool'+str(stage_no))(x)
        for module_no in range(modules_per_stage):
            x = fire_module(x, fire_id=module_idx, squeeze=squeeze, expand=expand)
            module_idx += 1

    x = Dropout(dropout, name='drop9')(x)
    x = Convolution2D(n_classes, (1, 1), padding='valid', name='topconv')(x)
    x = Activation('relu', name='relu_topconv')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)

    model = keras.Model(img_input, x)
    return model


"""

Based on https://github.com/arthurdouillard/keras-effnet/blob/master/effnet.py"""

from keras.models import Model
from keras.layers import *
from keras.activations import *

def get_post(x_in):
    #x = LeakyReLU()(x_in) # unsupported by STM32AI
    x = Activation('relu')(x_in)
    x = BatchNormalization()(x)
    return x

def get_block(x_in, ch_in, ch_out, kernel=3, downsample=2):
    x = Conv2D(ch_in,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False)(x_in)
    x = get_post(x)

    x = DepthwiseConv2D(kernel_size=(1, kernel), padding='same', use_bias=False)(x)
    x = get_post(x)
    x = MaxPool2D(pool_size=(downsample, 1),
                  strides=(downsample, 1))(x) # Separable pooling

    x = DepthwiseConv2D(kernel_size=(kernel, 1),
                        padding='same',
                        use_bias=False)(x)
    x = get_post(x)

    x = Conv2D(ch_out,
               kernel_size=(downsample, 1),
               strides=(1, downsample),
               padding='same',
               use_bias=False)(x)
    x = get_post(x)

    return x


def Effnet(input_shape, nb_classes, n_blocks=3,
            initial_filters=16, filter_growth=1.5, dropout=0.5, kernel=5, downsample=2,
            include_top='conv', weights=None):

    x_in = Input(shape=input_shape)
    x = x_in

    for block_no in range(n_blocks):
        filters_in = int(initial_filters*(filter_growth**block_no))
        filters_out = int(initial_filters*(filter_growth**(block_no+1)))
        x = get_block(x, filters_in, filters_out, kernel=kernel, downsample=downsample)

    if include_top == 'flatten':
        x = Flatten()(x)
        x = Dropout(dropout)(x)
        x = Dense(nb_classes, activation='softmax')(x)
    elif include_top == 'conv':
        # MobileNetv1 style
        x = GlobalAveragePooling2D()(x)
        shape = (1, 1, filters_out)
        x = Reshape(shape)(x)
        x = Dropout(dropout)(x)
        x = Conv2D(nb_classes, (1, 1), padding='same')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((nb_classes,), name='reshape_2')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

def build_model(frames=31, bands=60, channels=1, n_classes=10):
    shape = (bands, frames, channels)

    return Effnet(shape, nb_classes=n_classes)



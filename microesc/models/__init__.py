
from . import sbcnn, piczakcnn
from . import ldcnn, dcnn
from . import mobilenet, effnet, dense, squeezenet
from . import dmix, dilated
from . import skm, speech

families = {
    'piczakcnn': piczakcnn.build_model,
    'sbcnn': sbcnn.build_model,
    'ldcnn': ldcnn.ldcnn_nodelta,
    'dcnn': dcnn.dcnn_nodelta,
    'mobilenet': mobilenet.build_model,
    'effnet': effnet.build_model,
    'skm': skm.build_model,
    'squeezenet': squeezenet.build_model,
}

def build(settings):

    builder = families.get(settings['model'])

    options = dict(
        frames=settings['frames'],
        bands=settings['n_mels'],
        channels=settings.get('channels', 1),
    )

    # TODO: make more generic
    known_settings = [
        'conv_size',
        'conv_block',
        'downsample_type',
        'downsample_size',
        'filters',
    ]
    for k in known_settings:
        v = settings.get(k, None)
        if v is not None:
            if k == 'conv_size':
                k = 'pool'
            if k == 'conv_block':
                if v == 'dw' or v == 'dw_pw' or v == 'pw_dw_pw':
                    # FIXME: implement dwpw and pwdwpw
                    k = 'depthwise_separable'
                    v = True
                else:
                    continue
            if k == 'downsample_size':
                k = 'pool'
            if k == 'downsample_type':
                if v == 'stride':
                    k = 'use_strides'
                    v = True
                else:
                    continue
            if k == 'filters':
                k = 'kernels_start'

            options[k] = v

    print('o', options)
    model = builder(**options)
    return model


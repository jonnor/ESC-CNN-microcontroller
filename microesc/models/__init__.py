
from . import sbcnn, piczakcnn
from . import strided
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
    'strided': strided.build_model,
    'squeezenet': squeezenet.build_model,
}

def build(settings):

    builder = families.get(settings['model'])

    options = dict(
        frames=settings['frames'],
        bands=settings['n_mels'],
        channels=settings.get('channels', 1),
    )

    known_settings = [
        'conv_size',
        'conv_block',
        'downsample_size',
        'n_stages',
        'n_blocks_per_stage',
        'filters',
    ]
    for k in known_settings:
        v = settings.get(k, None)
        options[k] = v

    model = builder(**options)
    return model


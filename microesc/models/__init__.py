
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
    # LDCNN. filters=80, L=57, W=6
    # MobileNet. alpha, 
    known_settings = [
        'kernel', 'pool', 'kernels_start', 'fully_connected',
    ]
    for k in known_settings:
        v = settings.get(k, None)
        if v is not None:
            options[k] = v

    model = builder(**options)
    return model


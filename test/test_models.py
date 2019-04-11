
import pytest
import keras

from microesc import models
from microesc import settings

FAMILIES=list(models.families.keys())

@pytest.mark.parametrize('family', FAMILIES)
def test_models_basic(family):
    s = settings.load_settings({
        'model': family,
        'frames': 31,
        'n_mels': 60,
        'samplerate': 22050,
    })
    if family == 'sbcnn':
        s['downsample_size'] = (3, 2)
        s['conv_size'] = (3, 3)
    if family == 'strided':
        s['downsample_size'] = (3, 3)
        s['conv_size'] = (3, 3)
        s['conv_block'] = 'conv'
        s['filters'] = 12

    m = models.build(s)
    
    assert isinstance(m, keras.Model)


CONV_TYPES=[
    'conv',
    'depthwise_separable',
    'bottleneck_ds',
    'effnet',
]
@pytest.mark.parametrize('conv_type', CONV_TYPES)
def test_strided_variations(conv_type):

    s = settings.load_settings({
        'model': 'strided',
        'frames': 31,
        'n_mels': 60,
        'samplerate': 22050,
        'conv_block': conv_type,
        'filters': 20,
    })
    s['conv_size'] = (3, 3)
    s['downsample_size'] = (2, 2)

    m = models.build(s)
    assert isinstance(m, keras.Model)

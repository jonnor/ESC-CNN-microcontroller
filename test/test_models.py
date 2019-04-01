
import pytest
import keras

from microesc import models


FAMILIES=list(models.families.keys())

@pytest.mark.parametrize('family', FAMILIES)
def test_models_basic(family):
    s = {
        'model': family,
        'frames': 31,
        'n_mels': 60,
        'samplerate': 22050,
    }
    if family == 'sbcnn':
        s['kernel'] = (3, 3)

    m = models.build(s)
    
    assert isinstance(m, keras.Model)

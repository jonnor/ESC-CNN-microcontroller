
import os.path

import pytest
import keras

from microesc import models, stm32convert

out_dir = os.path.join(os.path.dirname(__file__), 'out')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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

    model_name = family+'.default'
    model_path = os.path.join(out_dir, model_name+'.hdf5')
    gen_path = os.path.join(out_dir, model_name)
    m.save(model_path)
    stats = stm32convert.generatecode(model_path, gen_path,
                                name='network', model_type='keras', compression=None)


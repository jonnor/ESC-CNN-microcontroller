
import os.path
import shutil

import numpy

from microesc import preprocess, urbansound8k, features, report

def test_precompute():

    settings = dict(
        feature='mels',
        samplerate=16000,
        n_mels=32,
        fmin=0,
        fmax=8000,
        n_fft=512,
        hop_length=256,
        augmentations=12,
    )

    dir = './pre2'
    if os.path.exists(dir):
        shutil.rmtree(dir)

    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/'))

    data = urbansound8k.load_dataset()
    urbansound8k.maybe_download_dataset(workdir)

    d = os.path.join(dir, features.settings_id(settings))
    expect_path = features.feature_path(data.iloc[0], d)
    assert not os.path.exists(expect_path), expect_path

    preprocess.precompute(data[0:4], settings, out_dir=d, verbose=0, force=True, n_jobs=2)

    assert os.path.exists(expect_path), expect_path
    

def test_grouped_confusion():
    cm = numpy.array([
       [82,  0,  3,  0,  0, 10,  0,  4,  1,  0],
       [ 3, 29,  0,  0,  0,  0,  1,  0,  0,  0],
       [ 4,  3, 37, 14,  4,  4,  0,  0,  2, 32],
       [ 5,  2,  5, 78,  4,  0,  0,  0,  0,  6],
       [23,  2,  4,  1, 55,  4,  2,  6,  3,  0],
       [ 9,  0,  0,  4,  3, 70,  0,  5,  1,  1],
       [ 0,  0,  0,  5,  0,  0, 27,  0,  0,  0],
       [ 0,  0,  2,  0,  1,  1,  1, 91,  0,  0],
       [ 9, 11,  9,  4,  0,  1,  0,  0, 46,  3],
       [ 1,  7,  7,  0,  7,  0,  0,  0,  3, 75]
    ])
    gcm, gnames = report.grouped_confusion(cm, report.groups)

    assert(numpy.sum(cm) == numpy.sum(gcm))
    assert(gnames[0] == 'social_activity')
    assert(gnames[3] == 'domestic_machines')

    expect_correct_social = (37+78+75)+(14+32)+(5+6)+(7+0) 
    # correct
    # + missclassified children playing as other social class
    # + missclassified dogbarks as other social class
    # + missclassified street music as other social class
    assert(gcm[0][0] == expect_correct_social), (gcm[0][0], expect_correct_social)

    # danger, only one class
    assert(gcm[3][3] == 82)

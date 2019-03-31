
import os.path
import shutil

from microesc import preprocess, urbansound8k, features

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
    

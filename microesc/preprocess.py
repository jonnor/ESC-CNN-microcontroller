
import os.path
import math
import shutil
import sys

import librosa
import numpy
import joblib

from . import urbansound8k, features, common


# TS,PS1 and PS1 from SalomonBello2016
def augmentations(audio, sr):
    
    ts  = [ 0.81, 0.93, 1.07, 1.23]
    ps = [ -2, -1, 1, 2, -3.5, -2.5, 2.5, 3.5 ]

    out = {}
    for stretch in ts:
        name = 'ts{:.2f}'.format(stretch)
        out[name] = librosa.effects.time_stretch(audio, stretch)

    for shift in ps:
        name = 'ps{:.2f}'.format(shift)
        out[name] = librosa.effects.pitch_shift(audio, sr, shift)

    return out

def precompute(samples, settings, out_dir, n_jobs=8, verbose=1, force=False):
    out_folder = out_dir

    def compute(inp, outp):
        sr = settings['samplerate']

        _lazy_y = None
        def load():
            nonlocal _lazy_y
            if _lazy_y is None:
                _lazy_y, _sr = librosa.load(inp, sr=sr)
                assert _sr == sr, _sr
            return _lazy_y

        if not os.path.exists(outp) or force:
            y = load()
            f = features.compute_mels(y, settings)
            numpy.savez(outp, f)

        if settings['augmentations']:

            paths = [ outp.replace('.npz', '.aug{}.npz'.format(aug)) for aug in range(12) ]
            exists = [ os.path.exists(p) for p in paths ]
            if not all(exists) or force:
                y = load()
                augmented = augmentations(y, sr).values()
                assert settings['augmentations'] == 12
                assert len(augmented) == settings['augmentations'], len(augmented)

                for aug, (augdata, path) in enumerate(zip(augmented, paths)):
                    f = features.compute_mels(augdata, settings)
                    numpy.savez(path, f)

        return outp
    
    def job_spec(sample):
        path = urbansound8k.sample_path(sample)
        out_path = features.feature_path(sample, out_folder)
        # ensure output folder exists
        f = os.path.split(out_path)[0]
        if not os.path.exists(f):
            os.makedirs(f)

        return path, out_path
        
    jobs = [joblib.delayed(compute)(*job_spec(sample)) for _, sample in samples.iterrows()]
    feature_files = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)



def parse(args):
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess audio into features')

    common.add_arguments(parser)
    a = parser.add_argument

    a('--archive', dest='archive_dir', default='',
        help='%(default)s')

    a('--jobs', type=int, default=8,
        help='Number of parallel jobs')
    a('--force', type=bool, default=False,
        help='Always recompute features')

    parsed = parser.parse_args(args)
    return parsed


def main():
    args = parse(sys.argv[1:])
    archive = args.archive_dir

    urbansound8k.default_path = os.path.join(args.datasets_dir, 'UrbanSound8K/')
    urbansound8k.maybe_download_dataset(args.datasets_dir)
    data = urbansound8k.load_dataset()
    settings = common.load_settings_path(args.settings_path)
    settings = features.settings(settings)
    features_path = os.path.join(args.features_dir, features.settings_id(settings))

    common.ensure_directories(features_path)
    precompute(data, settings, out_dir=features_path,
                verbose=2, force=args.force, n_jobs=args.jobs)

    if archive:
        print('Archiving as {}.zip'.format(features_path)) 
        shutil.make_archive(archive_path, 'zip', features_path) 


if __name__ == '__main__':
    main()

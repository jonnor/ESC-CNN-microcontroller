
import os.path
import math
import shutil
import sys

import librosa
import numpy
import joblib

from . import urbansound8k, features, common


# SalomonBello2016 experienced that only pitch shifts helped all classes
# Piczak2015 on the used class-dependent time-shift and pitch-shift
# https://github.com/karoldvl/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
def augment(audio, sr,
            pitch_shift=(-3, 3),
            time_stretch=(0.9, 1.1),):
    
    stretch = numpy.random.uniform(time_stretch[0], time_stretch[1]) 
    pitch = numpy.random.randint(pitch_shift[0], pitch_shift[1])
    aug = librosa.effects.time_stretch(librosa.effects.pitch_shift(audio, sr, pitch), stretch)

    return aug

def precompute(samples, settings, out_dir, n_jobs=8, verbose=1, force=False):
    out_folder = out_dir

    def compute(inp, outp):
        if os.path.exists(outp) and not force:
            return outp

        sr = sr=settings['samplerate']
        y, sr = librosa.load(inp, sr=sr)
        f = compute_mels(y, settings)
        numpy.savez(outp, f)

        for aug in range(settings['augmentations']):
            f = compute_mels(augment(y, sr=sr), settings)
            p = outp.replace('.npz', '.aug{}.npz'.format(aug))
            numpy.savez(p, f)

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

    urbansound8k.maybe_download_dataset(args.datasets_dir)
    data = urbansound8k.load_dataset()
    settings = common.load_experiment(args.experiments_dir, args.experiment)
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


import os.path
import uuid
import datetime
import sys

from . import common


template = """
apiVersion: batch/v1
kind: Job
metadata:
  name: mesc-{kind}-{name}
  labels:
    jobgroup: microesc-{kind}
spec:
  template:
    metadata:
      name: mesc-{kind}
      labels:
        jobgroup: microesc-{kind}
    spec:
      containers:
      - name: jobrunner
        image: {image}
        command: {command}
        securityContext:
          privileged: true
          capabilities:
            add:
              - SYS_ADMIN
        lifecycle:
          postStart:
            exec:
              command: ["gcsfuse", "-o", "nonempty", "--implicit-dirs", {bucket}, {mountpoint}]
          preStop:
            exec:
              command: ["fusermount", "-u", {mountpoint}]
        resources:
          requests:
            cpu: "1.3"
      restartPolicy: Never
"""


def array_str(a):
    m  = ', '.join([ '"{}"'.format(p) for p in a ])
    return '[ {} ]'.format(m)

def render_job(image, script, options, mountpoint, bucket):
    cmd = ["python3", "urbansounds/{}.py".format(script) ]

    for k, v in options.items():
        cmd += [ '--{}'.format(k), str(v) ]

    p = dict(
        image=image,
        kind=script,
        name=options['name'],
        command=array_str(cmd),
        bucket=bucket,
        mountpoint=mountpoint,
    )
    s = template.format(**p)
    return s

def generate_train_jobs(settings, jobs_dir, image, name, out_dir, mountpoint, bucket):

    folds = list(range(0, 9))

    settings['out'] = out_dir    
    for fold in folds:
        options = settings.copy()
        options['fold'] = fold
        options['name'] = "{}-fold{}".format(name, options['fold']) 

        s = render_job(image, 'train', options, mountpoint, bucket)

        job_filename = "train-{}.yaml".format(fold)
        out_path = os.path.join(jobs_dir, job_filename)
        with open(out_path, 'w') as out:
            out.write(s)

def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Train a model')

    common.add_arguments(parser)

    a = parser.add_argument


    a('--jobs', dest='jobs_dir', default='cloud/jobs',
        help='%(default)s')

    a('--bucket', type=str, default='jonnor-micro-esc',
        help='GCS bucket to write to. Default: %(default)s')

    a('--image', type=str, default='gcr.io/masterthesis-231919/base:12',
        help='Docker image to use')
    
    parsed = parser.parse_args(args)

    return parsed

def main():
    parsed = parse(sys.argv[1:])

    mountpoint = '/mnt/bucket'
    storage_dir = mountpoint+'/jobs'

    settings = common.load_experiment(args.experiments_dir, name)

    out = os.path.join(parsed.jobs_dir, name)

    t = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
    u = str(uuid.uuid4())[0:4]
    identifier = "-".join([name, t, u])

    if not os.path.exists(out):
        os.makedirs(out)

    generate_train_jobs(settings, jobs_dir, parsed.image,
                    identifier, storage_dir, mountpoint, parsed.bucket)
    print('wrote to', out)


if __name__ == '__main__':
    main()

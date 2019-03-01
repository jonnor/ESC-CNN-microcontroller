
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
      name: microesc-{kind}
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

def render_job(image, script, args, mountpoint, bucket):
    cmd = ["python3", "{}.py".format(script) ]

    for k, v in args.items():
        cmd += [ '--{}'.format(k), str(v) ]

    p = dict(
        image=image,
        kind=script,
        name=args['name'],
        command=array_str(cmd),
        bucket=bucket,
        mountpoint=mountpoint,
    )
    s = template.format(**p)
    return s

def generate_train_jobs(settings, jobs_dir, image, experiment, out_dir, mountpoint, bucket):

    t = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
    u = str(uuid.uuid4())[0:4]
    name = "-".join([experiment, t, u])

    folds = list(range(0, 9))
  
    for fold in folds:
        args = {
            'experiment': experiment,
            'models': out_dir,
            'fold': fold,
            'name': name+'-fold{}'.format(fold),
        }

        s = render_job(image, 'train', args, mountpoint, bucket)

        job_filename = "train-{}.yaml".format(fold)
        out_path = os.path.join(jobs_dir, job_filename)
        with open(out_path, 'w') as out:
            out.write(s)

def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Generate jobs')

    common.add_arguments(parser)

    a = parser.add_argument


    a('--jobs', dest='jobs_dir', default='./data/jobs',
        help='%(default)s')

    a('--bucket', type=str, default='jonnor-micro-esc',
        help='GCS bucket to write to. Default: %(default)s')

    a('--image', type=str, default='gcr.io/masterthesis-231919/base:18',
        help='Docker image to use')
    
    parsed = parser.parse_args(args)

    return parsed

def main():
    args = parse(sys.argv[1:])

    mountpoint = '/mnt/bucket'
    storage_dir = mountpoint+'/models'

    name = args.experiment
    settings = common.load_experiment(args.experiments_dir, name)

    out = os.path.join(args.jobs_dir, name)
    common.ensure_directories(out)

    generate_train_jobs(settings, out, args.image, name, storage_dir, mountpoint, args.bucket)
    print('wrote to', out)



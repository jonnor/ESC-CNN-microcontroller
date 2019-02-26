
import os.path
import uuid
import datetime

import yaml


template = """
apiVersion: batch/v1
kind: Job
metadata:
  name: microesc-{kind}-{name}
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

def generate_train_jobs(experiment_file, jobs_dir, image, name, out_dir, mountpoint, bucket):
    with open(experiment_file, 'r') as config_file:
        settings = yaml.load(config_file.read())

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


def main():

    # FIXME: support commandline arguments
    f = 'cloud/experiment.yaml'
    out = 'cloud/jobs/foo'
    image = 'gcr.io/masterthesis-231919/base:6'
    bucket = "jonnor-micro-esc"
    name = 'sbcnn.orig'

    mountpoint = '/mnt/bucket'
    out_dir = mountpoint+'/jobs'


    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') 
    u = str(uuid.uuid4())[0:8]
    identifier = "-".join([name, t, u])

    if not os.path.exists(out):
        os.makedirs(out)

    generate_train_jobs(f, out, image, identifier, out_dir, mountpoint, bucket)


if __name__ == '__main__':
    main()

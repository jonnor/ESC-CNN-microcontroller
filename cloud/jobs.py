
import os.path

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
      restartPolicy: Never
"""


def array_str(a):
    m  = ', '.join([ '"{}"'.format(p) for p in a ])
    return '[ {} ]'.format(m)

def render_job(image, script, options):
    cmd = ["python3", "urbansound/{}.py".format(script) ]

    for k, v in options.items():
        cmd += [ '--{}'.format(k), str(v) ]

    p = dict(
        image=image,
        kind=script,
        name=options['name'],
        command=array_str(cmd),
    )
    s = template.format(**p)
    print(s)
    return s

def generate_train_jobs(experiment_file, out_dir, image, name):
    with open(experiment_file, 'r') as config_file:
        settings = yaml.load(config_file.read())

    folds = list(range(0, 9))

    for fold in folds:
        options = settings.copy()
        options['fold'] = fold
        options['name'] = "{}-fold{}".format(name, options['fold']) 


        s = render_job(image, 'train', options)

        job_filename = "train-{}.yaml".format(fold)
        out_path = os.path.join(out_dir, job_filename)
        with open(out_path, 'w') as out:
            out.write(s)


def main():

    # FIXME: support commandline arguments
    f = 'cloud/experiment.yaml'
    out = 'cloud/jobs/foo'
    image = 'gcr.io/masterthesis-231919/base:3'
    name = 'foo'

    if not os.path.exists(out):
        os.makedirs(out)

    generate_train_jobs(f, out, image, name)


if __name__ == '__main__':
    main()

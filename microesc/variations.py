
import uuid
import os.path

import sklearn
import pandas

import models, stm32convert

def sbcnn_generator(n_iter=400, random_state=1):

    from sklearn.model_selection import ParameterSampler

    params = dict(
        kernel_t = range(3, 10, 2),
        kernel_f = range(3, 10, 2),
        pool_t = range(2, 5),
        pool_f = range(2, 5),
        kernels_start = range(16, 64),
        fully_connected = range(16, 128),
    )

    sampler = ParameterSampler(params, n_iter=n_iter, random_state=random_state)
    
    out_models = []
    out_total_params = []
    for p in sampler:
        s = {
            'model': 'sbcnn',
            'frames': 31,
            'n_mels': 60,
            'samplerate': 22050,
        }

        pool = p['pool_f'], p['pool_t'] 
        kernel = p['kernel_f'], p['kernel_t'] 
        for k, v in p.items():
            s[k] = v
        s['pool'] = pool
        s['kernel'] = kernel

        yield p, s


def generate_models():

    gen = sbcnn_generator()
    data = {
        'model_path': [],
        'gen_path': [],
        'id': [],
    }
    for out in iter(gen):
        model = None

        try:
            params, settings  = out
            model = models.build(settings.copy())
        except ValueError as e:
            print('Error:', e)
            continue


        # Store parameters
        for k, v in params.items():
            if data.get(k) is None:
                data[k] = []
            data[k].append(v)

        model_id = str(uuid.uuid4())
        out_dir = os.path.join('scan', model_id)
        os.makedirs(out_dir)

        model_path = os.path.join(out_dir, 'model.orig.hdf5')   
        out_path = os.path.join(out_dir, 'gen')     
   
        # Store model
        model.save(model_path)
        stats = stm32convert.generatecode(model_path, out_path,
                                  name='network', model_type='keras', compression=None)

        # Store model info
        data['model_path'].append(model_path)
        data['gen_path'].append(out_path)     
        data['id'].append(model_id)    

        for k, v in stats.items():
            if data.get(k) is None:
                data[k] = []
            data[k].append(v)

    df = pandas.DataFrame(data)
    return df

def main():

    df = generate_models()
    df.to_csv('scan.csv')


if __name__ == '__main__':
    main()

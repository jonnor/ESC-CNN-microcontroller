
import math
import os.path
import sys

import tensorflow as tf
import numpy
import pandas
import keras

from . import stm32convert

def is_training_scope(scope):
    patterns = ('/random_uniform', '/weight_regularizer', '/dropout_', '/dropout/', 'AssignMovingAvg')

    is_training = False
    for t in patterns:
        if t in scope:
            is_training = True

    return is_training

# Profiling calculation based on
# https://stackoverflow.com/questions/43490555/how-to-calculate-a-nets-flops-in-cnn
# and https://stackoverflow.com/a/50680663/1967571
# @build_func - function returning a Keras TensorFlow model
def analyze_model(build_func, input_shapes, n_classes):

    from tensorflow.python.framework import graph_util
    import tensorflow.python.framework.ops as ops
    from tensorflow.compat.v1.graph_util import remove_training_nodes
    from tensorflow.python.tools import optimize_for_inference_lib

    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with tf.Session(graph=g) as sess:
        keras.backend.set_session(sess)

        base = build_func()

        inputs = []
        for shape in input_shapes:
            input_shape = [1] + list(shape)
            inp = tf.placeholder(tf.float32, input_shape)
            inputs.append(inp)
        model = base(inputs)

        # Get number of parameters
        opts = tf.profiler.ProfileOptionBuilder().trainable_variables_parameter() 
        opts['output'] = 'none'
        params_stats = tf.profiler.profile(g, run_meta=run_meta, cmd='scope', options=opts)
        params = {}
        for scope in params_stats.children:
            #print('s', scope)
            params[scope.name] = scope.total_parameters

        # Get number of flops
        flops = {}
        opts = tf.profiler.ProfileOptionBuilder().float_operation()
        opts['output'] = 'none'
        flops_stats = tf.profiler.profile(g, run_meta=run_meta, cmd='scope', options=opts)
        for scope in flops_stats.children:
            flops[scope.name] = scope.total_float_ops

        return flops, params

def layer_info(model):

    df = pandas.DataFrame({
        'name': [ l.name for l in model.layers ] ,
        'type': [ l.__class__.__name__ for l in model.layers ],
        'shape_in': [ l.get_input_shape_at(0)[1:] for l in model.layers ],
        'shape_out': [ l.get_output_shape_at(0)[1:] for l in model.layers ],
    })
    df['size_in'] = df.shape_in.apply(numpy.prod)
    df['size_out'] = df.shape_out.apply(numpy.prod)
    return df

def stm32layer_sizes(stats):
    activation_types = set(['_output_array', '_output_in_array', '_output_out_array'])
    weight_types = set(['_weights_array', '_bias_array', '_scale_array'])
    array_types = activation_types.union(weight_types)
    
    def lazy_add(d, key, value):
        if d.get(key, None) is None:
            d[key] = 0
        d[key] += value
    
    activations = {}
    weights = {}
    
    for name, size in stats['arrays'].items():
        
        known = False
        for suffix in array_types:
            if name.endswith(suffix): 
                layer_name = name.rstrip(suffix)
                out = activations if suffix in activation_types else weights
                lazy_add(out, layer_name, size)
                known = True

        assert known, 'Unknown array {}'.format(name)

    layers = set(activations.keys()).union(set(weights.keys())) 
          
    df = pandas.DataFrame({
        'activations': [ activations.get(n, math.nan) for n in layers  ],
        'weights': [ weights.get(n, math.nan) for n in layers ],
    }, dtype='int', index=list(layers))
        
    return df

def check_model_constraints(model, max_ram=64e3, max_maccs=4.5e6*0.72, max_flash=512e3):

    out_dir = './out' # FIXME: use tempdir

    model_path = os.path.join(out_dir, 'model.hd5f')
    out_path = os.path.join(out_dir, 'gen')
    model.save(model_path)

    stats = stm32convert.generatecode(model_path, out_path,
                                  name='network', model_type='keras', compression=None)

    layers = layer_info(model)
    sizes = stm32layer_sizes(stats)
    combined = layers.join(sizes, on='name', how='inner')

    def check(val, limit, message):
        assert val <= limit, message.format(val, limit)

    check(stats['flash_usage'], max_flash, "FLASH use too high: {} > {}")
    check(stats['ram_usage_max'], max_ram, "RAM use too high: {} > {}")
    check(stats['maccs_frame'], max_maccs, "CPU use too high: {} > {}")

    del stats['arrays']

    return stats, combined


def main():

    # XXX: analysis window overlap / voting not taken into account
    # FIXME: take length of frame (in seconds) into account

    sample_rate=44100
    window_stride_ms=10

    def build_speech_tiny():
        return speech.build_tiny_conv(input_frames=frames, input_bins=bands, n_classes=10)

    # TODO: output window size (ms), and input size (ms)
    models = {
        'SB-CNN': (sbcnn.build_model, [(128, 128, 1)]),
#        'Piczak': (piczakcnn.build_model, [(60, 41, 2)]),
#        'SKM': (skm.build_model, [(40,173,1)]),
#        'DilaConv': (dilated.dilaconv, [(64, 41, 2)]),
#        'D-CNN': (dilated.dcnn, [(60, 31, 1), (60, 31, 1)]),
#        'LD-CNN': (dilated.ldcnn, [(60, 31, 1), (60, 31, 1)]),
#        'Dmix-CNN': (dmix.build_model, [(128, 128, 2)]),
        #'speech-tiny': build_speech_tiny,
#        'cnn-one-fstride4': (speech.build_one, [(40, 61, 1)]),
    }

    model_params = {}
    model_flops = {}
    model_stats = { name: analyze_model(build, shape, n_classes=10) for name, (build, shape) in models.items() }
    for name, stats in model_stats.items():
        flops, params = stats

        inference_flops = { name: v for name, v in flops.items() if not is_training_scope(name) }
        total_flops = sum(inference_flops.values()) 
        total_params = sum(params.values())

        model_params[name] = total_params
        model_flops[name] = total_flops

        print(name)
        print('Total: {:.2f}M FLOPS, {:.2f}K params'.format(total_flops/1e6, total_params/1e3))
        print('\n'.join([ "\t{}: {} flops".format(name, v) for name, v in inference_flops.items()] ))
        print('')
        print('\n'.join([ "\t{}: {} params".format(name, v) for name, v in params.items()] ))
        print('\n')

    print('p', model_params)
    print('f', model_flops)

    #spec_flops = fft_splitradix(fft_length)*n_frames
    # TODO: take into account mel-filtering
    # TODO: take into account log    

    #speedup = spec_flops/cnn_flops
    #print(speedup, cnn_flops, spec_flops)


if __name__ == '__main__':
    main()

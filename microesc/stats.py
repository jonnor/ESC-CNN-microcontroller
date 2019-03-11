
import math
import os
import sys

import tensorflow as tf
import numpy
import keras.layers

from .models import sbcnn, speech, dilated, skm, piczakcnn, dmix

def fft_splitradix(N):
    return 4*N*math.log(N,2) - (6*N) + 8


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

def next_power_of_two(x):
  """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def logmel_raw_compare(sample_rate=44100, window_stride_ms=10):

    # XXX: analysis window overlap / voting not taken into account
    # FIXME: take length of frame (in seconds) into account


    def build_speech_tiny():
        return speech.build_tiny_conv(input_frames=frames, input_bins=bands, n_classes=10)

    # TODO: output window size (ms), and input size (ms)
    models = {
        'SB-CNN': (sbcnn.build_model, [(128, 128, 1)]),
        'Piczak': (piczakcnn.build_model, [(60, 41, 2)]),
        'SKM': (skm.build_model, [(40,173,1)]),
        'DilaConv': (dilated.dilaconv, [(64, 41, 2)]),
        'D-CNN': (dilated.dcnn, [(60, 31, 1), (60, 31, 1)]),
        'LD-CNN': (dilated.ldcnn, [(60, 31, 1), (60, 31, 1)]),
        'Dmix-CNN': (dmix.build_model, [(128, 128, 2)]),
        #'speech-tiny': build_speech_tiny,
        'cnn-one-fstride4': (speech.build_one, [(40, 61, 1)]),
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
    logmel_raw_compare()


import math
import os

import tensorflow as tf
import keras.backend as K

import models_keras

def fft_splitradix(N):
    return 4*N*math.log(N,2) - (6*N) + 8



# Profiling calculation based on
# https://stackoverflow.com/questions/43490555/how-to-calculate-a-nets-flops-in-cnn
# @model - a Keras TensorFlow model
def profile_flops(build_func):

    run_meta = tf.RunMetadata()
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        model = build_func()

        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        # params.total_parameters
        
        return flops.total_float_ops

def next_power_of_two(x):
  """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def logmel_raw_compare(sample_rate=16000, window_stride_ms=10):

    window_length_ms = 2*window_stride_ms
    fft_length = next_power_of_two((sample_rate * (window_length_ms)/1000))
    n_frames = 1000/window_length_ms

    # XXX: assumes 1second windows, processing without overlap?
    def build():
        net = models_keras.build_aclnet_lowlevel(sample_rate)
        return net

    cnn_flops = profile_flops(build)
    
    spec_flops = fft_splitradix(fft_length)*n_frames
    # TODO: take into account mel-filtering
    # TODO: take into account log    

    speedup = spec_flops/cnn_flops
    print(speedup, cnn_flops, spec_flops)


if __name__ == '__main__':
    logmel_raw_compare()

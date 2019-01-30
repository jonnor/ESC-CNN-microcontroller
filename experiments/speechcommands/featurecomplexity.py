
import math
import os

import tensorflow as tf
import keras.backend as K
import numpy

import models_keras

def fft_splitradix(N):
    return 4*N*math.log(N,2) - (6*N) + 8



# Profiling calculation based on
# https://stackoverflow.com/questions/43490555/how-to-calculate-a-nets-flops-in-cnn
# @model - a Keras TensorFlow model
def profile_flops(build_func):

    from keras.objectives import categorical_crossentropy

    run_meta = tf.RunMetadata()
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)

        #print('o', outout)

        inp = tf.placeholder(tf.float32, (1,44100,1))
        labels = tf.placeholder(tf.float32, (1, 1))

        model = build_func(inp)
        preds = model.output
        #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        #model.fit(y=outout)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
        #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        numpy.random.seed(1)
        outout = numpy.random.choice([0,1], size=(1,1))
        data = numpy.ones(shape=(1,44100,1))

        sess.run(preds, feed_dict={inp: data})

        #with sess.as_default():
        #    train_step.run(feed_dict={inp: data, labels: outout, K.learning_phase(): 1})

        #model.input = inp
        #model.output = targets
        #foo = model.predict(data)
        #print('foo', foo.shape)

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


def logmel_raw_compare(sample_rate=44100, window_stride_ms=10):

    window_length_ms = 2*window_stride_ms
    fft_length = next_power_of_two((sample_rate * (window_length_ms)/1000))
    n_frames = 1000/window_length_ms

    # XXX: assumes 1second windows, processing without overlap?
    def build(tensor):
        net = models_keras.build_aclnet_lowlevel(sample_rate, input_tensor=tensor)
        return net

    cnn_flops = profile_flops(build)
    
    spec_flops = fft_splitradix(fft_length)*n_frames
    # TODO: take into account mel-filtering
    # TODO: take into account log    

    speedup = spec_flops/cnn_flops
    print(speedup, cnn_flops, spec_flops)


if __name__ == '__main__':
    logmel_raw_compare()

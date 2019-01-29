
## Onepager

Audio Classification on microcontrollers

Jon Nordby <jonnord@nmbu.no>

PICTURE
Board. Coin as size ref
Sensor costs `<1000` NOK

Microphone
Microcontroller
Radio transmitter

Sound -> Inference -> Classification -> Transmission

Application example: Industrial monitoring.

    Motor off
    Operation normal
    Maintenance needed
    Failure

## Experiment notes

Converting Tensorflow model to Keras.
Need to manually write Keras model, and the load weights.
https://stackoverflow.com/questions/44466066/how-can-i-convert-a-trained-tensorflow-model-to-keras/53638524#53638524

### Speech commands

Reproducing existing TensorFlow tutorials.

mfcc40
`--preprocess=mfcc --how_many_training_steps=20000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv` reaches about 80% val accuracy. Final test 77.8%. Final test 78.3%.

average40
`--preprocess=average --how_many_training_steps=30000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv`
fails to reach more than 50% accuracy...

logmel32
`preprocess=logmel --feature_bin_count 32 --how_many_training_steps=20000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv`
Reaches about 80% val accuracy after 7000 epochs.
Trains faster than MFCC, performs much better than average.
Final test 

`--preprocess=mfcc --feature_bin_count=13 --how_many_training_steps=20000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv` ? MFCC13 

`--preprocess=mfcc --model_architecture=low_latency_svdf --how_many_training_steps=100000,3500 --learning_rate=0.01,0.005`
Should reach 85%.

Main model should reach between 85% and 90%.

Is SVDF used/usable for CNNs? Seems to work well for DNN and RNN

FastGRNN does amazingly on Speech commands. How does it do on ECS-10/50 or Urbanset 8k?


## Methodology

- Select a general purpose microcontroller. Having a specific amount of storage,RAM and CPU.

STM32 family spans a large range.
Mid-range. STM32L4 low-power (ARM Cortex M4F+).
High perf. STM32F4, STM32F7.
Low-perf. STM32F1/STM32L1
Support for standard microphone,LogMel,CNNs out-of-the-box 
Note: most usecases require (wireless) connectivity.
Can use separate chip for demo?
SensorTile devkit has Bluetooth chip included.

- Select open datasets for audio classification
Acoustic event detection,
Acoustic noise source classification,
Acoustic Scene classification,
Keyword spotting
Speech command
Should have at least one binary classification, and one multi-label. Maybe detection/segmentation
Should have one with short events, and one with longer effects "scene" etc

- Select a couple of baseline methods
With reference results available on 
Ideally with open code for easily reproducability.

- Create and tests hypothesis for approaches to take for making more efficient models


Power efficiency. Measurement: Current consumption. Proxy: CPU inference time 
Cost efficient. Measurement: Microcontroller cost. Proxy: RAM,CPU requirements

Not All Ops Are Created Equal!, https://arxiv.org/abs/1801.04326
Found up to 5x difference in throughput/energy between different operations.

## Ideas for optimization

Approaches

- Compute less.
- Compute more efficiently
- Model compression
- Space/compute tradeoffs

## Compute less

Use softmax instead of Dense layer(s) to combine features

Use 1D convolution instead of Dense to flatten feature vector

Use 1D convolutions instead of 2D.

Stacked in layers. Or 2d separable.
DS-CNN for KWS by ARM had good results (on MFCC).

Scattering transform might be good feature for 1D conv? melspectrogram might not be?


Hypothesis: A tree-based classifier is more CPU/storage efficient than FC/conv as last part of CNN
Test: Replace last layers with tree-based classifier, check perf vs storage/execution
Test: Use knowledge distillation to a soft decision tree (Hinton 2017)
Some support in Adaptive Neural Trees, https://arxiv.org/abs/1807.06699. Good CIFAR10,MINST perf with few parameters.
and Deep Neural Decision Forests.

Hypothesis: Using raw audio convolution filters instead of computing STFT/melspec/MFCC can save considerable compute
Test: Find how much percent of time is used for MFCC feature calculation versus classifier 
Estimate how deep filters can be before FFT is beneficial
Does it help to initialize initial convolutions as well-behaved filters?
Can we perform a greedy search for filters?

Hypothesis: Optimizing execution path across an entire forest/GBM can reduce compute time
Test: Check literature for existing results
How to reduce redundancies across nodes without causing overfitting
Can one identify critical nodes which influence decisions a lot, and should be done first
Can one know when a class has gotten so much support that no other nodes need to be evaluated
Can many similar nodes be combined into fatter ones?
Probabalistic
Intervals
Test: Count how often features are accessed in forest/GBM. Plot class distributions wrt feature value (histogram) and thresholds

Hypothesis: On-demand computation of features can save significant amount of time.
Test: Use decision_path() to determine how often features are accessed per sample
Using GradientBoostedTrees/RandomForest/ExtraTrees as classifier, pulling in convolutions as needed.
Memoization to store intermediate results.
Flips dataflow in the classifier from forward to backward direction

Hypothesis: Pruning spectrogram field-of-view can reduce computations needed

Reduce from top (high frequency)
Reduce from bottom (low frequency)
Reduce in middle?

Use LIME to visualize existing networks and get some idea of possibility of reduction


Can we prune convolutions inside network?
Prune kernels. Prunt weights
Estimate importance, eliminate those without much contributions
Or maybe introduce L1 regularization?

Architecture search.
MnasNet: Towards Automating the Design of Mobile Machine Learning Models
https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html
> Formulate a multi-objective optimization problem that aims to achieve both high accuracy and high speed,
> and utilize a reinforcement learning algorithm with a customized reward function to find Pareto optimal solutions 
> With the same accuracy, our MnasNet model runs 1.5x faster
> than the hand-crafted state-of-the-art MobileNetV2, and 2.4x faster than NASNet.

How to choose optimal hyperparameters for mel/spectrogram calculation
Frame length (milliseconds)
Frame overlap (percent/milliseconds)
Number of bands. fmin, fmax
Can a sparse FFT save time?
Challenge: Interacts with model, especially convolution sizes


## Not so relevant

* DCASE2018 Task 5.
Domestic activities. 
10 second segments.
9 classes.
From 4 separate microphone arrays (in single room).
Each array has 4 microphones.



## Compute more efficiently

Winograd convolutional kernels

## Space-compute tradeoffs

Can weights of convolution kernels be decomposed and expressed as a combination of smaller pieces?
BinaryCmd does this, with some success on Keyword spotting.




## Model compression

Lots of existing work out there.

Quantized weights
Quantized activations
Pruning channels
Sparsifying weights

Custom On-Device ML Models with Learn2Compress 
https://ai.googleblog.com/2018/05/custom-on-device-ml-models.html
Uses pruning, quantization, distillation and joint-training
CIFAR-10 94x smaller than NASNet, perf drop 7%


## References


Tang2018AnEA
https://arxiv.org/pdf/1711.00333.pdf
> Study the power consumption of a family of convolutional neural networks for keyword spotting on a Raspberry PI.
> We find that both number of parameters and multiply operations are good predictors of energy usage,
> although the number of multiplies is more predictive than the number of model parameters


RETHINKING THE VALUE OF NETWORK PRUNING
After pruning, retraining from scratch is more efficient than keeping original weights.
Pruning can be seen as a type of architecture search.
! references state-of-the-art pruning methods for CNNs
Network pruning dates back to, Optimal Brain Damage (LeCun et al., 1990)


Learning from Between-class Examples for Deep Sound Recognition
https://openreview.net/forum?id=B1Gi6LeRZ
Data augmentation technique designed for audio,
quite similar to mixup.

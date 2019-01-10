
# Machine Learning on Microcontrollers
Initial research happening in [EmbeddedML](./embeddedml),
to be distilled into a Master Thesis here.


## Title

Efficient audio classification using general-purpose microcontrollers

## Problem statement

How to make (cost,power) efficient audio classification
for real-time use on microcontroller?

Scope:

Focused on Convolutional Neural Networks as a base/framework

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
Stacked in layers. Or 2d separable

On-demand computation of features.
Using GradientBoostedTrees/RandomForest/ExtraTrees as classifier, pulling in convolutions as needed.
Memoization to store intermediate results.
Flips dataflow in the classifier from forward to backward direction

Avoid spectrogram feature calculation.
Use filters on time-domain directly.
Does it help to initialize 

Can we prune the s
Reduce from top (high frequency)
Reduce from bottom (low frequency)

Can we prune convolutions inside network?
Estimate importance, eliminate those without much contributions
Or maybe introduce L1 regularization?


How to choose optimal hyperparameters for mel/spectrogram calculation
Frame length (milliseconds)
Frame overlap (percent/milliseconds)
Number of bands. fmin, fmax
Can a sparse FFT save time?
Challenge: Interacts with model, especially convolution sizes

## Compute more efficiently

Winograd convolutional kernels

## Space-compute tradeoffs

Can weights of convolution kernels be decomposed and expressed as a combination of smaller pieces?


## Model compression

Lots of existing work out there.

Quantized weights
Quantized activations



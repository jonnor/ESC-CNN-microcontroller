
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

### Urbansound8k

### Dilated model.
Reaching 64%-69% val accuracy on 35k samples, with 32,32,64,64 kernels.
Significantly higher than train, indicates dropout is working well?
But after epoch 3/4 val_loss starts going higher than acc, sign of overfitting.
Due to overcapacity?

Training about 7 minute per epoch of 35k samples.

32,32,32,32. Also seems to start overfitting after 68% accuracy at epoch 5, but a bit less severe.
Combined val accuracy at 65%. Test accuracy at 57% :(
Almost all mis-classifications are into the 'drilling' class. Unknown why??

! warning, the windowing function was changed between train and test...

#### SB-CNN
Trains much faster than Dilated. Approx 1 minute per epoch of 35k samples.
First version seems to peak at 60% validation during train.
Afterwards windowed validation is up to 63% and test is 65/67%. 
More resonable confusion matrix than Dilated, less being classified as Drilling,
but still more on testing set versus training set.

With augmentations, seems to also peak at 59.5% validation during train.
Testing accuracy also does not improve. Overregularized?



#### Validation.

Worker setup time. 5 minutes
Preprocessing 15 minutes.
Time per epoch. 10 minutes. 10 epochs++, 100 minutes
Folds. 10x folds.
Est: 10x120 minutes, 20 hours per model...

Job inputs.
Settings.
Fold number.

Job outputs.
Best trained model. HDF5
Predictions (probabilities) for all samples using best model.



### Speech commands

Reproducing existing TensorFlow tutorials.

mfcc40
`--preprocess=mfcc --how_many_training_steps=20000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv` reaches about 80% val accuracy. Final test 77.8%. Final test 78.3%.

average40
`--preprocess=average --how_many_training_steps=30000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv`
fails to reach more than 50% accuracy...

logmel32
`preprocess=logmel --feature_bin_count 32 --how_many_training_steps=20000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv`
Reaches about 80% val accuracy after 7000 epochs, about 84% total.
Trains faster than MFCC, and performs much better than average.
Final test accuracy = 83.5%.

MFCC13?
`--preprocess=mfcc --feature_bin_count=13 --how_many_training_steps=20000,6000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv`

SVDF
`--preprocess=mfcc --model_architecture=low_latency_svdf --how_many_training_steps=100000,3500 --learning_rate=0.01,0.005`
Should reach 85%.


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

# Hypotheses

## Raw audio instead of spectrogram input

Hypothesis: Using raw audio convolution filters instead of computing STFT/melspec/MFCC can save considerable compute

Tests:

- Find how much percent of time is used for feature calculation versus classifier 
- Test 1D CNN in comparison. ACL

Ideas:

- Does it help to initialize initial convolutions as well-behaved filters?
- Can we perform a greedy search for filters?

## Tree-based CNN backend

### Hypothesis: A tree-based classifier is more CPU/storage efficient than FC/conv as last part of CNN
Test: Replace last layers with tree-based classifier, check perf vs storage/execution
Test: Use knowledge distillation to a soft decision tree (Hinton 2017)
Some support in Adaptive Neural Trees, https://arxiv.org/abs/1807.06699. Good CIFAR10,MINST perf with few parameters.
and Deep Neural Decision Forests.

### Hypothesis: Optimizing execution path across an entire forest/GBM can reduce compute time
Test: Check literature for existing results
How to reduce redundancies across nodes without causing overfitting
Can one identify critical nodes which influence decisions a lot, and should be done first
Can one know when a class has gotten so much support that no other nodes need to be evaluated
Can many similar nodes be combined into fatter ones?
Probabalistic
Intervals
Test: Count how often features are accessed in forest/GBM. Plot class distributions wrt feature value (histogram) and thresholds

### Hypothesis: On-demand computation of features can save significant amount of time.
Test: Use decision_path() to determine how often features are accessed per sample
Using GradientBoostedTrees/RandomForest/ExtraTrees as classifier, pulling in convolutions as needed.
Memoization to store intermediate results.
Flips dataflow in the classifier from forward to backward direction


### Spectrogram pruning

Hypothesis: Pruning spectrogram field-of-view can reduce computations needed

- Reduce from top (high frequency)
- Reduce from bottom (low frequency)
- Try subsampling filters on input. Equivalent to reducing filterbank bins?

How to test

- Use *LIME* to visualize existing networks to get some idea of possibility of reduction
- Use *permutation feature importance* on spectrogram bins to quantify importance of each band


### Stacked 1D instead of 2D

Hypothesis: Stacked 1D convolutions instead of 2D are more compute efficient

Ideas:

- Scattering transform might be good feature for 1D conv? Or MFCC.
Melspectrogram might not be, since information spreads out over bands.

Related:

- DS-CNN for KWS by ARM had good results with depthwise-separable CNN (on MFCC).


## Other

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




## Compute more efficiently

Winograd convolutional kernels

## Space-compute tradeoffs

Can weights of convolution kernels be decomposed and expressed as a combination of smaller pieces?
BinaryCmd does this, with some success on Keyword spotting.

## Model compression

Lots of existing work out there.

- Quantized weights
- Quantized activations
- Pruning channels
- Sparsifying weights

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


Better Machine Learning Models with Multi Objective Optimization
https://www.youtube.com/watch?v=oSLASLV4cTc
applying multi-object optimization to feature selection
Better options than greedy SFS,SBS.
Suggests using Evolutionary Algorithms instead.
Example on the sonar dataset.
Regularized risk. Empirical risk (accuracy/metric) plus tradeoff * structural risk (model complexity).
Problem: how to choose tradeoff
Using multi-objective optimization to optimize them at the same time. Accuracy and number of parameters.
Formulated as a Pareto front.
Non-dominated sorting. Builds up population by first selecting individuals that dominate others.
Gives the best results, and can inspect selection.

Can be used for unsupervised/clustering also, which is classically hard.
When clustering feature selection tends to push number of features down.
Because smaller dimensions condenses value space.
When using multi-objective optimization, maximize the number of features.

What about greedy algorithms with random restarts/jumps?



### Automatic Environmental Sound Recognition: Performance versus Computational Cost
2016.

Intended target platform ARM Cortex M4F.
Compares performance of different classifiers on ESC task, using different classifiers.
MFCC input. 13 bands, with deltas.
GMM,SVM,k-NN.

! no CNNs present
! only theoretical N-operations shown, not actual runtime

Evaluated on Baby Cry and Smoke Alarm datasets. Binary classification tasks.
DNNs gave best performance, and perf/computation.

! gives adds/multiplies formulas for each classifier type

### Design aspects of acoustic sensor networks for environmental noise monitoring
September 2016.
https://www.sciencedirect.com/science/article/pii/S0003682X16300627

Categorizes ESC sensors into 4 categories,
 ased on Hardware Costs, Scalability, Flexibility, Accuracy.

Evaluated different usecases.
Noise annoyance, Illegal firework detection/localization, Noise reason monitoring for tram passings. 

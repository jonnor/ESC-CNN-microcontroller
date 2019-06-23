
## Onepager

Audio Classification on microcontrollers

Jon Nordby <jonnord@nmbu.no>

PICTURE
Board. Coin as size ref
Sensor costs

Microphone
Microcontroller
Radio transmitter

Sound -> Inference -> Classification -> Transmission


## Experiment notes

Converting Tensorflow model to Keras.
Need to manually write Keras model, and the load weights.
https://stackoverflow.com/questions/44466066/how-can-i-convert-a-trained-tensorflow-model-to-keras/53638524#53638524

### Urbansound8k

In [@chu2009environmental] the authors conducted a listening test and found that 4 seconds
were sufficient for subjects to identify environmental sounds
with 82% accuracy.

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

dilated32

  MACC / frame: 38 844 758
  ROM size:     110.91 KBytes
  RAM size:     477.00 KBytes (Minimum: 477.00 KBytes)

! did not train correctly on Google Colab?

### Strided model

For DS-5x5 12, going from 0.5 dropout to 0.25 increases perf from 65% to 72%

python train.py --model strided --conv_block depthwise_separable --epochs 100 --downsample_size=2x2 --filters 12 --dropout 0.25

### Aggregation
Low-pass filter over consequtive frames?
Exponential Moving Average?

## Testing

Jackhammer
https://annotator.freesound.org/fsd/explore/%252Fm%252F03p19w/
https://freesound.org/people/Mark_Ian/sounds/131918/

Dog bark
https://annotator.freesound.org/fsd/explore/%252Fm%252F0bt9lr/
http://freesound.org/s/365053




## Kubernetes

One-time setup

    gcloud config set project masterthesis-231919
    gcloud config set compute/zone europe-north1-a

    gcloud auth application-default login
    
Setup gcloud bucket in Kubernetes

    https://github.com/maciekrb/gcs-fuse-sample

Unpacking a zip of files to GCS bucket mounted with FUSE was incredibly slow.
Over 1 second per file, average 100kB size.

Accessing files in the mount seems better, under 100ms to read file.
But local access is sub 1ms.

rsync from GCS is 80MB/s for large files.
Maybe zip + streaming unpacking is way to go?
Should get feature-set with 5 augmentations down to 2-3 minutes bootstrapping.

ZIP cannot generally be unzipped in streaming fashion.
tar.xz archives on the other hand can?

A single .npz file with all the features would avoid zipping.
But needs a transformation from when preprocessing anyway.

With n1-highcpu-2, SB-CNN on 32mels, 16kHz, 1 sec window takes approx 2min per batch of 100 samples.
2 hours total for 50 epochs.

SB-CNN 32mel 16kHz 1 sec 50% vote overlap had much lower validation performance than testset.
Across most folds.

SB-CNN 128mel 3 sec 16kHz 50% vote overlap on the other hand was very similar, as expected.

### 16 kHz

Can reach at least 72% val

    python train.py --settings experiments/16k30_256hop.yaml --conv_size=3x3  --downsample_size=2x4 --conv_block=depthwise_separable

    Epoch 28/50
    75/75 [==============================] - 53s 711ms/step - loss: 1.7626 - acc: 0.3819 - val_loss: 1.4058 - val_acc: 0.6210

    Epoch 00028: saving model to ./data/models/unknown-20190424-1453-ed2a-fold0/e28-v1.41.t1.76.model.hdf5
    voted_val_acc: 0.6816

    Epoch 31/50
    75/75 [==============================] - 53s 706ms/step - loss: 1.7569 - acc: 0.3848 - val_loss: 1.3921 - val_acc: 0.6348

    Epoch 00031: saving model to ./data/models/unknown-20190424-1453-ed2a-fold0/e31-v1.39.t1.76.model.hdf5
    voted_val_acc: 0.6999

    Epoch 00048: saving model to ./data/models/unknown-20190424-1453-ed2a-fold0/e48-v1.32.t1.69.model.hdf5
    voted_val_acc: 0.7113
    Epoch 49/50
    75/75 [==============================] - 52s 689ms/step - loss: 1.6900 - acc: 0.4004 - val_loss: 1.2917 - val_acc: 0.6337

    Epoch 00049: saving model to ./data/models/unknown-20190424-1453-ed2a-fold0/e49-v1.29.t1.69.model.hdf5
    voted_val_acc: 0.6735
    Epoch 50/50
    75/75 [==============================] - 52s 690ms/step - loss: 1.6830 - acc: 0.4051 - val_loss: 1.2508 - val_acc: 0.6731

    Epoch 00050: saving model to ./data/models/unknown-20190424-1453-ed2a-fold0/e50-v1.25.t1.68.model.hdf5
    voted_val_acc: 0.7205


However does max 63% with this strided model ?

    [jon@jon-thinkpad thesis]$ python train.py --settings experiments/16k30_256hop.yaml --conv_size=3x3  --downsample_size=2x4 --conv_block=depthwise_separable --model strided


    Epoch 49/50
    75/75 [==============================] - 34s 460ms/step - loss: 1.7501 - acc: 0.3760 - val_loss: 1.4669 - val_acc: 0.5940

    Epoch 00049: saving model to ./data/models/unknown-20190424-1602-8cd3-fold0/e49-v1.47.t1.75.model.hdf5
    voted_val_acc: 0.6323



### Effects of different overlap in voting

Quick test on SB-CNN16k 30mels, fold0, validation
0.1, acc 0.6666666666666666
0.5, acc 0.6746849942726232
0.9, acc 0.6758304696449027

```
After models have been chosen with 0.5 overlap:

python report.py --results data/results/overlap0/ --run 20190408-0629 --out data/results/overlap0/

res
   experiment  test_acc_mean  maccs_frame
0          1       0.693748   10185806.0
1          2       0.703305    3180954.0
2          0       0.715651     530162.0

python report.py --run 20190408-0629

res
   experiment  test_acc_mean  maccs_frame
0          1       0.708084   10185806.0
1          2       0.713262    3180954.0
2          0       0.718439     530162.0
```



### STM32Ai

arm_rfft_fast_init_f32 called for every column

Preprocessing. 1024 FFT. 30 mels. 8 cols.
Before. MelColumn 8/16 ms. Approx 1-2 ms per col
After. Same!.
Reason: The function does not compute the twiddle factors, just set up pointer to pregenerated table

Missing function for window functions.
https://github.com/ARM-software/CMSIS_5/issues/217



## Ideas for optimization

Approaches

- Compute less.
- Compute more efficiently
- Model compression
- Space/compute tradeoffs

# Hypotheses

### Stacked 1D conv instead of 2D

Hypothesis: Stacked 1D convolutions instead of 2D are more compute efficient

Ideas:

- Scattering transform might be good feature for 1D conv? Or MFCC.
Melspectrogram might not be, since information spreads out over bands.

Related:

- DS-CNN for KWS by ARM had good results with depthwise-separable CNN (on MFCC).

### Strided convolutions

fstride-4 worked well on keyword spotting
Could maybe be applied to LD-CNN?

### Fully Convolutional

Using Global Average Pooling instead of fully-connected

### Dilated convolutions

LD-CNN with two heads fails in STM32AI.
Probably multi-input is not implemented?
Or one of the more rare operations, like Add

LD-CNN with one head loads in STM32AI

DilaConv also loads, though has way too much RAM/MACCS with 32,32,64,46 kernels.


## Raw audio instead of spectrogram input

Hypothesis: Using raw audio convolution filters instead of computing STFT/melspec/MFCC can save considerable compute

Tests:

- Find how much percent of time is used for feature calculation versus classifier 
- Test 1D CNN in comparison. ACL

Ideas:

- Does it help to initialize initial convolutions as well-behaved filters?
- Can we perform a greedy search for filters?


Is this strided convolution on raw audio
more computationally efficent than STFT,log-mel calculation?

LLF from ACLNet: 1.44k params, 4.35 MMACS. 2 conv, maxpool.
1.28 second window. 64x128 output. Equivalent to 64 bin, 10ms skip log-mel?
Can it be performed with quantizied weights? 8 bit integer. SIMD.
Would be advantage, FFT is hard to compute in this manner...
Advantage, potential offloading to CNN co-processor

Calc mult-add from model.
Tensorflow, https://stackoverflow.com/questions/51077386/how-to-count-multiply-adds-operations

https://dsp.stackexchange.com/questions/9267/fft-does-the-result-of-n-log-2n-stand-for-total-operations-or-complex-ad
def fft_splitradix(N):
    return 4*N*math.log(N,2) - (6*N) + 8

Could one use teacher-student / knowledge distillation to pre-train 1D conv on raw audio?
Previously raw audio conv have been quite different than logmel, advantageous together.
Maybe this allows training a version similar to logmel, which can still be executed with convolutions,
and combined together for higher perf?


## Convolutional Random Forest 

Cluster spectrogram input using spherical k-means clustering
Use these as candidate kernels on different spectrogram locations
Attempt patches can be sampled from anywhere, or from specific frequency bands
Attempt to use small kernels (ex 5x5)
with strides (2x, 4x)
Attempt to use spatially separable kernels (5x1 -> 1x5)
Can be done at multiple scales of the spectrogram
Use to evaluate as splits in Random Forest
Using memoization

Related
CATBoost.
"CatBoost: gradient boosting with categorical features support"
Handles categorical variables at training time.
Constructs combinations of categorical variables. Greedy
Also does numerical-categorical combinations in same manner

CATBoost. Oblivious trees as base predictor.
Splitting criterion is same across entire level of tree.

### A tree-based classifier is more CPU/storage efficient than FC/conv as last part of CNN
Test: Replace last layers with tree-based classifier, check perf vs storage/execution
Test: Use knowledge distillation to a soft decision tree (Hinton 2017)
Some support in Adaptive Neural Trees, https://arxiv.org/abs/1807.06699. Good CIFAR10,MINST perf with few parameters.
and Deep Neural Decision Forests.

### Optimizing execution path across an entire forest/GBM can reduce compute time
Test: Check literature for existing results
How to reduce redundancies across nodes without causing overfitting
Can one identify critical nodes which influence decisions a lot, and should be done first
Can one know when a class has gotten so much support that no other nodes need to be evaluated
Can many similar nodes be combined into fatter ones?
Probabalistic
Intervals
Test: Count how often features are accessed in forest/GBM. Plot class distributions wrt feature value (histogram) and thresholds

### On-demand computation of features can save significant amount of time.
Test: Use decision_path() to determine how often features are accessed per sample
Using GradientBoostedTrees/RandomForest/ExtraTrees as classifier, pulling in convolutions as needed.
Memoization to store intermediate results.
Flips dataflow in the classifier from forward to backward direction


## Spectrogram pruning

Hypothesis: Pruning spectrogram field-of-view can reduce computations needed

- Reduce from top (high frequency)
- Reduce from bottom (low frequency)
- Try subsampling filters on input. Equivalent to reducing filterbank bins?

How to test

- Use *LIME* to visualize existing networks to get some idea of possibility of reduction
- Use *permutation feature importance* on spectrogram bins to quantify importance of each band
- Make the STFT-mel filter trainable, with L1 regularization
- Use a fully convolutional CNN with support for different size inputs, in order to estimate feature importance?
Ideally without retraining, or possibley with a bit of 
- Can we use a custom layer in the front with weights for the different frequency bands, and L1 regularization?
Maybe something like a dense layer from n_bands_in -> n_bands_out. And try higher and higher compression.


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


Gammatone spectrograms are defined as linear filters
Could be used to avoid FFT?
Approximated with an IIR.
https://github.com/detly/gammatone/blob/master/gammatone/gtgram.py


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



# Power draw

perftest firmware
3.3V
Measured 35 mA when executing network.
30mA when not executing
?? much higher than expected.


STM32L4SystemPower PDF

All numbers at 1.8V?

RUN1. 10.5mA @ 80 Mhz
LPRun. 270uA @ 2Mhz. 
LPSleep. 80uA @ 2Mhz. SAI/ADC still active




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

dilated32

  MACC / frame: 38 844 758
  ROM size:     110.91 KBytes
  RAM size:     477.00 KBytes (Minimum: 477.00 KBytes)

! did not train correctly on Google Colab?

#### SB-CNN
Trains much faster than Dilated. Approx 1 minute per epoch of 35k samples.
First version seems to peak at 60% validation during train.
Afterwards windowed validation is up to 63% and test is 65/67%. 
More resonable confusion matrix than Dilated, less being classified as Drilling,
but still more on testing set versus training set.

With augmentations, seems to also peak at 59.5% validation during train.
Testing accuracy also does not improve. Overregularized?

  MACC / frame: 3 199 214
  ROM size:     197.60 KBytes
  RAM size:     27.29 KBytes (Minimum: 27.29 KBytes)

ROM and RAM is OK.
A lot of MACCs... Might be approx 0.5 second inference time?

#### Multiple instance on SB-CNN
Quite fast. 1 minute per epoch.

! the train acc is average of all batches in epoch.
So when a lot of learning happens within an epoch, val_acc will be higher
! calling fit_generator() or model.compile() does not reset training!

When using GlobalMeanPooling and default RMSprop
With batchsize=10, starts overfitting at 30% val
With batchsize=25, starts overfitting at 35% val
With batchsize=50, seems to start overfitting at 45%
With batchsize=100, seems to start overfitting at 47%

When using GlobalMaxPooling
With batchsize=100, hit 61% val_acc.
Progress seems a bit noisy, often overfitting with val_acc near 50%
Struggling a lot with children playing and street music

Ideas for trainingset expansion techniques for MIM?
Swap windows internally in sample. No change with GlobalAveragePooling?
Replace window with one from another same-class samples
Take a window from sample of same class, mix it into our window? 

"Adaptive pooling operators for weakly labeled sound event detection". 2018
Proposes auto-pool, mixing min/max/average pooling with learned parameters.
But still has a hyperparamter lambda that must be tuned.
Evaluted on URBAN-SED, a Sound Event Detection dataset based on Urbansound8k.
Said to apply generally to MIM problems.

SB-CNN 10MACC reaches 75% with lr=0.001. 
0.003 and 0.005 only reaches 72-73%

effnet 0.7MACC reaches 66% with lr=0.01
69% with lr=0.005
But RAM usage too high!
When using strides=(2,2) in first layer, only gets to 51%

#### DenseNet

dropout=0.0, depth=10, block=3, growth=12, pooling='avg' (45k parameters)
seems to reach around 65% on validation during training.
However is overfitting, large gap to train loss.

Takes almost 10 minutes per epoch for 20k samples

growth=10 makes val_acc drop to 25% !!
dropout=0.5 makes drop to 50% val_acc, still overfitting ??
growth=16 also seems to overfit after 1 epoch. 50% val

Probably this network is way to deep for our dataset.

With depth=4, block=3 trains fast. 1 min per epoch.
But seems hard to get val_acc over 54%
!! when depth=4, dropouts are not present?

With depth=7,blocks=2,dropout=0.5 trains to 60% val_acc
pooling='avg', dropout=0.5, growth=30, reduction=0.5
trains to 62% val_acc with 0.0001 learning rate.
Does not look to overfit, but still only did 55% on testset.
A lot of misclassifications into children_playing 

!!! DenseNet models faile to validate in STM32CubeMXAI.
No explanation why 

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

### Model choosing

With SGD (default options).

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1737-5c84-fold0/e11-v1.34.t1.63.model.hdf5 
acc 0.718213058419244
train val_acc: 0.6432

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1737-5c84-fold0/e17-v1.27.t1.50.model.hdf5
acc 0.6849942726231386
train val_acc: 0.6192

! better val_loss, but poorer prediction on whole set!
Need to go make model selection on windowed performance 

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1737-5c84-fold0/e23-v1.34.t1.42.model.hdf5 
acc 0.6643757159221076
train val_acc: 0.6134

Another run


[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2115-8a3d-fold0/e14-v1.34.t1.70.model.hdf5 
acc 0.6918671248568156
val_acc 0.64

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2115-8a3d-fold0/e16-v1.35.t1.67.model.hdf5 
acc 0.6827033218785796
val_acc 0.617

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2115-8a3d-fold0/e18-v1.29.t1.63.model.hdf5 
acc 0.6758304696449027
val_acc 0.649

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2115-8a3d-fold0/e20-v1.28.t1.59.model.hdf5
acc 0.693012600229095

600/600 [==============================] - 47s 79ms/step - loss: 1.5868 - acc: 0.4394 - val_loss: 1.2534 - val_acc: 0.6504
[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2115-8a3d-fold0/e21-v1.25.t1.59.model.hdf5 
acc 0.7101947308132875
val_acc 0.65



With SGD(lr=0.1, decay=0.1/30)

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2158-bc8a-fold0/e16-v1.28.t1.53.model.hdf5 
acc 0.6815578465063001

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2158-bc8a-fold0/e22-v1.29.t1.51.model.hdf5 
acc 0.6884306987399771
val_acc 0.624


Another run

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2219-d651-fold0/e21-v1.30.t1.49.model.hdf5 
acc 0.6506300114547537

Very stable around val_loss 1.3 / val_acc 0.60. From epoch 7 - 30 almost


With SGD(momentum=0.9, nesterov=True)

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1816-32d3-fold0/e02-v1.36.t1.69.model.hdf5
acc 0.6334478808705613

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1816-32d3-fold0/e06-v1.32.t1.38.model.hdf5 
acc 0.6827033218785796


With SGD(momentum=0.9, nesterov=True) and reducing to 30k samples / epoch

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1828-e66d-fold0/e02-v1.43.t1.85.model.hdf5
acc 0.7010309278350515

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1828-e66d-fold0/e03-v1.32.t1.71.model.hdf5 
acc 0.6701030927835051

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1828-e66d-fold0/e06-v1.28.t1.51.model.hdf5
acc 0.6964490263459335

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1828-e66d-fold0/e07-v1.28.t1.46.model.hdf5 
acc 0.6941580756013745

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1828-e66d-fold0/e08-v1.20.t1.43.model.hdf5 
acc 0.7353951890034365
test_acc: 0.725

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-1828-e66d-fold0/e10-v1.23.t1.38.model.hdf5 
acc 0.7079037800687286

From another run

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model data/models/sbcnn16k30-20190302-2003-8c69-fold0/e03-v1.35.t1.72.model.hdf5
acc 0.6987399770904925
val_acc 0.6396

However 2 more times one failed to yield models significantly above val_acc 0.60

Going up to batchsize=100 (from 50)

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2022-5e53-fold0/e05-v1.33.t1.63.model.hdf5 
acc 0.6907216494845361
valacc: 0.62

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2022-5e53-fold0/e10-v1.38.t1.42.model.hdf5 
acc 0.6735395189003437

But another time failed to find any above val_acc 0.60


With SGD(lr=0.1, decay=0.1/(epochs*2), momentum=0.3, nesterov=True)

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2303-974b-fold0/e15-v1.30.t1.45.model.hdf5 
acc 0.6849942726231386

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2303-974b-fold0/e16-v1.29.t1.44.model.hdf5 
acc 0.6758304696449027
val_acc 0.62

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2303-974b-fold0/e17-v1.32.t1.43.model.hdf5 
acc 0.6781214203894617


With SGD(lr=0.001, momentum=0.95, nesterov=True)

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2335-829b-fold0/e09-v1.32.t1.63.model.hdf5
acc 0.7101947308132875
val_acc 0.65

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2335-829b-fold0/e20-v1.30.t1.41.model.hdf5 
acc 0.6517754868270332
val_acc 0.61

Another run

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2359-16a9-fold0/e10-v1.37.t1.63.model.hdf5 
acc 0.6781214203894617

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2359-16a9-fold0/e20-v1.30.t1.42.model.hdf5 
acc 0.6735395189003437

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2359-16a9-fold0/e22-v1.28.t1.39.model.hdf5 
acc 0.6941580756013745
val_acc 0.644

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2359-16a9-fold0/e26-v1.23.t1.35.model.hdf5 
acc 0.6941580756013745
val_acc 0.66

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2359-16a9-fold0/e29-v1.20.t1.34.model.hdf5 
acc 0.6987399770904925
val_acc 0.65

With keras.optimizers.SGD(lr=0.001, momentum=0.90, nesterov=True)

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190303-0025-2398-fold0/e12-v1.38.t1.72.model.hdf5
acc 0.6827033218785796



With AdaDelta()

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2052-6d84-fold0/e03-v1.36.t1.61.model.hdf5
acc 0.6769759450171822

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2057-57e0-fold0/e08-v1.33.t1.42.model.hdf5 
acc 0.6792668957617412

[jon@jon-thinkpad thesis]$ python3 test.py --experiment sbcnn16k30 --model ./data/models/sbcnn16k30-20190302-2057-57e0-fold0/e11-v1.35.t1.34.model.hdf5 
acc 0.6632302405498282

Avoids overfitting semi-well. Train/val scores get close and stay like that for 5 epochs.



With SB-CNN on 20k, 60 mels. 200k parameters

Epoch 00038: saving model to ./data/models/ldcnn20k60-20190306-1624-e15e-fold0/e38-v1.38.t1.63.model.hdf5
voted_val_acc: 0.7113
Epoch 39/50
80/80 [==============================] - 30s 371ms/step - loss: 1.6213 - acc: 0.4241 - val_loss: 1.4305 - val_acc: 0.5880


Under same circumstances seems to be really hard to get LD-CNN above 0.63




Seems reasonably easy to train to 0.60 window perf and 0.65 voted perf.
Going to 0.64 window / 0.70 voted possible but unpredictable?

Sensitive to hyperparameter changes...
Do hyperparameter search? With goal of reaching 0.70 voted reliably
Random sampling.
At least over alpha
Maybe batch size?
Maybe also beta

Good explanations of momentum in SGD. With visual+interactive plots, physical correspondence
https://distill.pub/2017/momentum/

> When the problem’s conditioning is poor, the optimal alpha α is approximately twice that of gradient descent,
> and the momentum term is close to 1.
? So set beta β as close to 1 as you can, and then find the highest alpha α which still converges.

! With windowed prediction, out-of-fold versus in-fold is quite different. Suggests overfitting?
acc 0.6712485681557846
acc 0.8524774774774775


### STM32Ai

arm_rfft_fast_init_f32 called for every column

Preprocessing. 1024 FFT. 30 mels. 8 cols.
Before. MelColumn 8/16 ms. Approx 1-2 ms per col
After. Same!.
Reason: The function does not compute the twiddle factors, just set up pointer to pregenerated table

Missing function for window functions.
https://github.com/ARM-software/CMSIS_5/issues/217



Not All Ops Are Created Equal!, https://arxiv.org/abs/1801.04326
Found up to 5x difference in throughput/energy between different operations.


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



### Design aspects of acoustic sensor networks for environmental noise monitoring
September 2016.
https://www.sciencedirect.com/science/article/pii/S0003682X16300627

Categorizes ESC sensors into 4 categories,
 ased on Hardware Costs, Scalability, Flexibility, Accuracy.

Evaluated different usecases.
Noise annoyance, Illegal firework detection/localization, Noise reason monitoring for tram passings. 


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







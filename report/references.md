
# Methods

### UNSUPERVISED FEATURE LEARNING FOR URBAN SOUND CLASSIFICATION
2015.

Using spherical-k-means to learn single layer of convolutions.

Reaches 72% accuracy.
With further augmentation can reach 75% accuracy

log-melspectrogram input.
44100 Hz sample rate. window size of 23 ms (1024 at 44.1 kHz), hop size 1024

Patches were 40 band tall. 8 frames long.
codebook size k=2000
+ Random Forest


## ENVIRONMENTAL SOUND CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORKS
https://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
2015.
PiczakCNN
Tested on Urbansound, ESC-10 and ESC-50.

22050Hz sample rate, 1024 window, 512 hop, 60 mels

5 variation of models. Baseline, short/long segments, majority/probability voting.
Short segments: 41 frames, approx 950 ms.
Long segments: 101 frames, approx 2.3 seconds.
Average accuracy from 69% to 72%. Best model, LP long+probability.
Parameters not specified. Estimated 25M ! (almost all from 5000,5000 FC layers) 

### Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
Justin Salamon and Juan Pablo Bello.
November 2016.
https://arxiv.org/pdf/1608.04363.pdf
SB-CNN. 73% without augmentation, 79% with data augmentation.
3-layer convolutional, using 5x5 conv and max pooling.
References models PiczakCNN 72% avg acc and SKM 72% avg acc.
Parameters not specified. Estimated 444k


### Deep Convolutional Neural Network with Mixup for Environmental Sound Classification
https://link.springer.com/chapter/10.1007/978-3-030-03335-4_31
November, 2018.

log-melspectrogram, gammatone compared
Uses mixup and data augmentation.
Uses stacked 1-D convolutions in some places
(3,7), (3,5), (3,1), (3,1), (1,5), (1,5)

83.7% on UrbanSound8k with Gammatones.

window size of 1024, hop length of 512 and 128 bands,128 frames (1.5sec)

SGD with learning rate decay. Initial rate 0.1. Nesterov, 0.9 momentum
Batch 200 samples randomly selected without replacement
200/300 epochs training
mean voting

### Learning environmental sounds with end-to-end convolutional neural network
EnvNet

https://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK.pdf
When combined with log-mel, gives 6.5% improvement.

Learned CNN frontend outputs 40x150 feature map.
Tests different filter sizes for low level features. For 44.1kHz.
Find 8 long performs best. With 2 conv layers.
! graphs showing frequency response of the learned layers.
When filter indexes are sorted, has similar shape to mel-scale.

On ESC-50. Input lengths of 1-2.5 seconds perform similarly. 60% with raw only

logmel only. 58.9%
logmel,logmel-delta. 66.5%

EnvNet raw. 64%
EnvNet logmel,raw. 69.3%
EnvNet logmel,logmeldelta,raw. 71%

Uses a 1 second window.
Randomly chosen at training time.
Mean voting over windows. 0.2second stride 

Urbansound8k 66.7% (from EnvNet2 paper)

### Learning from Between-class Examples for Deep Sound Recognition
https://openreview.net/forum?id=B1Gi6LeRZ
Feb 2018.

EnvNet-v2 is like EnvNet but with
44.1 kHz instead of 16kHz,
13 layers instead of 7.

Without special learning, 69.1% on Urbansound8k
Using between-class learning and strong augmentation, got
78.3% on Urbansound8k
84.9% on ESC-50, 91.4% on ESC-10


### Dilated Convolution Neural Network with LeakyReLU for Environmental Sound Classification
https://ieeexplore.ieee.org/document/8096153
Xiaohu Zhang, Yuexian Zou, Wei Shi
2015

D-CNN
Same features as Piczak
22050Hz sample rate, 1024 window, 512 hop
31 frames for Urbansound8k

81.9% accuracy on Urbansound8k
ReLu 81.2%

log-mel + delta log-mel
Time stretching, noise additon

### Environmental sound classification with dilated convolutions
https://www.sciencedirect.com/science/article/pii/S0003682X18306121
December, 2018

DilaConv

Notes that a dilated convolution with striding can replaces max pooling,
and give bigger field-of-view.

Kernel size 3x3. Dilation rate 2, striding 2x2.
Several numbers of dilated convolutions and conv layers were attempted.
Best performance was found with 4 layers, where 1-3 layers were DilaConv.

! very hard to read the results, only in graphs, and across multiple groups
! not compared with plain striding
! not compared with plain max pooling 
! claims better performance than existing, but uses very weak baselines/comparisons
? no mention of data augmentation applied?

The models struggled particularly with Air Conditioner class,
only 45% accuracy. 


log-mel + delta-logmel
sample rate 22500
window size of 1024, hop length of 512
64 n-mels
41 frames
Global Average Pooling, Softmax output
SGD 0.0001, 0.9 Nesterov momentum
500 epochs training

! good images in the article.

### LD-CNN: A Lightweight Dilated Convolutional Neural Network for Environmental Sound Classification
2018.
http://epubs.surrey.ac.uk/849351/1/LD-CNN.pdf

LD-CNN
Based on D-CNN, but 50 times smaller, only looses 2% points accuracy

Early layers use 1D stacked convolutions.
Then dilated convolution in the middle.

log-mel + delta-logmel
time-stretching augmentation
60 mels
31 frames on Urbasound
101 frames on ESC50

SDG 0.01 learning rate, 0.9 momentum
mean voting between windows

Model size 2.05MB.
79% Urbansound.
66% ESC-50.

References multiple other lightweight ESC models testing.
?? Claims DenseNet performs well at 390.3KB size.
But their reference does not support this, it is just a general intro to DenseNet concept.


The first layer is nearly full height/frequency (57 of 60 bands),
and processed with stride 1, giving 4 values out.
Since each filter covers basically entire spectrum,
this seems basically equivalent to having full-height frames, just 4x as many of them?
The different filters will have to learn potentially the same patterns,
just at different locations inside the kernel?
Can one reduce the height to say half of band height, and use strides to get 4-8.
Means combination of low + high frequency patterns will have be learned by later layers...

`cnn-one-fstride4` in Google Keyword Spotting paper does essentially this.



### Very deep convolutional neural networks for raw waveforms
2016

Reproduction in Keras
https://github.com/philipperemy/very-deep-convnets-raw-waveforms

### WSNet: COMPACT AND EFFICIENT NETWORKS WITH WEIGHT SAMPLING
https://arxiv.org/abs/1711.10067
Xiaojie Jin, Yingzhen Yang

2018.

CNN trained on raw audio.
8 layers

Compared on UrbanSound8k and ESC-50.
70.5% average acc on UrbandSound8k.

2 model variations evaluated, plus quantized versions.
520K and 288K parameters.

#### AclNet: efficient end-to-end audio classification CNN
https://arxiv.org/abs/1811.06669
November, 2018
Intel

> AclNet gets its inspirations for efficient computations from MobileNet
Depthwise separable convolutions

Using data augmentation and mixup got 5% improvement
a=0.1/0.2 for mixup
Input length of 1.5 sevonds. 1-2 seconds worked OK.

Low-level features (LLF).
Two layers of 1D strided convolution as FIR decimation filterbank.
64 channels x 128 frames (with 10 ms frame rate).

Then a VGG type architecture as backend. Conv->Conv->Maxpool, 4 blocks
Uses MobileNet style width multiplier to adjust complexity,
controls number of channels in layers

Evaluated on ESC-50
Depthwise-Separable @44kHz. 81.75% accuracy, 155k parameters, 49M multiply-adds/second. 
Standard Convolutio @44kHz, 82.20% accuracy, 84k parameters, 131M multiply-adds/second.

16kHz topped at 80.9%, just below human perf.
75% acc with 3M MACC/s using 16kHz
75% acc took 20M MACC/s with 44.1kHz


### WaveMsNet, Learning Environmental Sounds with Multi-scale Convolutional Neural Network
https://arxiv.org/abs/1803.10219
March, 2018

Uses 1-d convolutions on raw audio in frontend.
Tests small,medium,large receptive field, and a model with all.
Multi-scale outperforms single by 2-3% absolute.

When only using raw audio. 70% on ESC-50, 88% on ESC-10
When combined with log-mel features, 79.10% on ESC-50. 93.75% on ESC-10.


### Audio representation for environmental sound classification using convolutional neural networks
https://lup.lub.lu.se/student-papers/search/publication/8964345
2018.
Master thesis. 66 pages total.
30 pages theory, 6+14 pages on experiment.

Used SBCNN as the base model. "for promise in embedded systems"
spectrograms as base feature.
Tried mel and linear spectrograms.
Tried downsampling datarate.

Used mean-normalization per patch.
Used 128 mel-frequency bands.

Evaluated on ESC-50.
Best model mel-spectrogram. Table 4.1
M 2048 75 44.1 top1=74.70% top3=88.35%

75% overlap better than 50%.
.. Down to acc=66% with 32/16kHz sample rate.
! did not change N_fft with different frequencies, so temporal resolution differs also

Future work.
Raw audio as input feature.
Number of mel-specrogram points.
Different filterbanks.

### Multi-Channel Convolutional Neural Networks with Multi-Level Feature Fusion for Environmental Sound Classification
January 2019.
https://www.researchgate.net/publication/329569964_Multi-channel_Convolutional_Neural_Networks_with_Multi-level_Feature_Fusion_for_Environmental_Sound_Classification_25th_International_Conference_MMM_2019_Thessaloniki_Greece_January_8-11_2019_Proceedi

MC-DCNN.
Stacking 1D layers with very small filters (except first layer).
Multi-level fusion. Multiple CNN heads at different resolutions, features are concatenated.
Fully convolutional, using 1D CNN backend with global-average-pooking.

Number of parameters ?

! Table 5 has good overview of performance, relative to other models.
On raw audio showing performance similar to M18,EnvNet2
Urbansound8k 73.6%.
ESC-50 71.1±0.8  ESC-10 84.1±0.7

Fusion with 3 heads performs better.

### Look, Listen and Learn
August, 2017.

Unsupervised learning of audio+image embeddings.
Using log-spectrogram input.
Output audio: 512 dimensional vector.

ESC-50. 79.3%

### LEARNING FILTER BANKS USING DEEP LEARNING FOR ACOUSTIC SIGNALS
Shuhui Qu.

Based on the procedure of log Mel-filter banks, we design a filter bank learning layer.
Urbansound8K dataset, the experience guided learning leads to a 2% accuracy improvement.


### Automatic Environmental Sound Recognition: Performance versus Computational Cost
2016.
https://ieeexplore.ieee.org/document/7515194/
https://arxiv.org/abs/1607.04589.pdf
Sacha Krstulović and Mark D. Plumbley

Intended target platform ARM Cortex M4F.
Compares performance of different classifiers.
GMM,SVM,k-NN.
! no CNNs present

MFCC input. 13 bands, with deltas.
Evaluated on Baby Cry and Smoke Alarm datasets, binary classification tasks.
DNNs gave best performance, and perf/computation.

Only theoretical N-operations shown, not actual runtime
! gives adds/multiplies formulas for each classifier type. Nice



### Not All Ops Are Created Equal!
https://arxiv.org/abs/1801.04326
[@lai2018not]
Found up to 5x difference in throughput/energy between different operations.


# Efficient CNNs for Keyword Spotting

### Convolutional Neural Networks for Small-footprint Keyword Spotting 
https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
2015

Designs small CNNs
Keeping number of multiples the same, at 500 M
Number of params around 50-100k

40 log-mel spectrogram
25 ms frames
10 ms shift
32 frame windows = 310 ms
! 1 frame shifts = 96% overlap

1 layer or 2 layer CNNs, followed by 2 dense layers
Testing pooling and striding alternatives
Getting very good results for striding in time and striding in frequency


## Hello Edge: Keyword Spotting on Microcontrollers
https://arxiv.org/abs/1711.07128
2017

!uses MFCC
Reviews many deep learning approaches. DNN, CNN, RNN, CRNN, DS-CNN.
Depthwise Separable Convolutional Neural Network (DS-CNN)
provides the best accuracy while requiring significantly lower memory and compute resources.
Considering 3 different sizes of networks, bound by NN memory limit and ops/second limits.
Small= 80KB, 6M ops/inference, 94.5% accuracy.
8-bit weights and 8-bit activations, with KWS running at 10 inferences per second.
Each inference – including memcopy, MFCC feature extraction and DNN execution – takes about 12 ms.
10x inferences/second. Rest sleeping = 12% duty cycle.

Explained in
"How to Achieve High-Accuracy Keyword Spotting on Cortex-M Processors"
https://community.arm.com/processors/b/blog/posts/high-accuracy-keyword-spotting-on-cortex-m-processors.

Running on STM32F746G-DISCO.
DSCNNL model gives 0.83/84.6% on Kaggle leaderboard
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data

## Compressing Deep Neural Networks using a Rank-Constrained Topology
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

Implemented in Tensorflow
Ex `low_latency_svdf` with 750K parameters and 750kFLOPS.
Reaching 0.85 acc on Speech Commands dataset

## Studying the Effects of Feature Extraction Settings on the Accuracy and Memory Requirements of Neural Networks for Keyword Spotting
https://ieeexplore.ieee.org/abstract/document/8576243

Compares MFCC feature extraction settings. Compares RAM/ROM use.
10-40 MFCC bands. 10-40ms size.
! GRU used 0.66 KB RAM with 314KB FLASH reaching 92%.
DS-CNN used 62KB RAM with 161KB FLASH reaching 93.5%

## On-the-fly deterministic binary filters for memory efficient keyword spotting applications on embedded devices
https://dl.acm.org/citation.cfm?id=3212731

BinaryCmd makes represents weights as a combination of predefined orthogonal binary basis.
that can generate convolutional filters very efficiently on-the-fly.
Deter-ministic Binary Filters, DBF.
Orthogonal variable spreading factor, OVSF
Using MFCC features.
Claims state of the art results (over Hey Edge) for models under 3MOPs and 30kB of model size.
! but CRNN at 3.3MOP has much higher performance
! smaller variation of CRNN nor DS-CNN not tested...

## FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network
https://www.microsoft.com/en-us/research/publication/fastgrnn-a-fast-accurate-stable-and-tiny-kilobyte-sized-gated-recurrent-neural-network/
Evaluated on Google Speech Command Set, both 30 and 12 class. Clips are 1 second
12 class: Smallest model 5.5KB, 92% acc, 242 ms on Cortex M0+ @ 48Mhz.
Using Log mel spectrograms, 32 mels, 25ms window, 10ms stride



## Stochastic Adaptive Neural Architecture Search for Keyword Spotting
https://arxiv.org/abs/1811.06753
2018

Proposes SANAS (Stochastic Adaptive Neural Architecture Search)

Designing multiple architectures with different complexity,
switching automatically at runtime to use simpler models to reduce CPU time 

Evaluated on Speech Command Set

cnn-trad-fpool3 used 120-130 MFLOPS/frame for 72.8% correct,
Their solution 40M for 80% correct and higher match for matched or slightly better perf.

Code at http://github.com/TomVeniat/SANAS



# Efficient CNNs

## Image models

MobileNet
Figure 4 shows log linear dependence between accuracy and computation.


MobilenetV2 [@Mobilenetv2] (2018) builds on Mobilenets and
adds a pointwise convolution ... into the convolutional blocks.
Ratio between the size of the input bottleneck and the inner size as the expansion ratio.
Shortcut connections between bottlenecks.

Max activtions size 200K float16, versus 800K for MobileNetV1 and 600K for ShuffleNet.
Smallest network at 96x96 with 12M mult-adds, 0.35 width.
Performance curve very similar to ShuffleNet.


ReLU6 as the non-linearity. Designed for with low-precision computation (8 bit fixed-point). y = min(max(x, 0), 6).
Combined with SSDLite, gives similar object detection performance as YOLOv2 at 10% model size and 5% compute.
200ms on Pixel1 phone using TensorFlow Lite.

[@SquuezeNext]
Notes inefficiency of depthwise-separable convolution in terms of hardware performance,
due to its poor arithmetic intensity (ratio of compute to memory operations). REF Williams2009


CondenseNet: An Efficient DenseNet using Learned Group Convolutions.
https://arxiv.org/abs/1711.09224
More efficient than MobileNet and ShuffleNets.

[FD-MobileNet: Improved MobileNet with a Fast Downsampling Strategy](https://arxiv.org/abs/1802.03750). February 2018.
1.1x inference speedup over MobileNet. And 1.82x over ShuffleNet.


## Explanations

[3 Small But Powerful Convolutional Networks](https://towardsdatascience.com/3-small-but-powerful-convolutional-networks-27ef86faa42d).
Explains MobileNet, ShuffleNet, EffNet.
Visualizations of most important architecture differences, and the computational complexity benefits.

[Why MobileNet and Its Variants (e.g. ShuffleNet) Are Fast](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d).

Covers MobileNet, ShuffleNet, FD-MobileNet.
! Explains the convolution variants used visually.
Pointwise convolution (conv1x1), grouped convolution, depthwise convolution.

https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

## Depthwise conv

Depthwise convolutions special case of grouped convolutions. n_groups == n_channels

MobileNet
https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69
Explains the width multiplier alpha,
and resolutions multiplier

MobileNet got close to InceptionV3 results with 1/8 the parameters and multiply-adds

Xception uses depthwise-separable to get better performance over InceptionV3
https://arxiv.org/abs/1610.02357v3
https://vitalab.github.io/deep-learning/2017/03/21/xception.html
https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568
! no activation in Xception block. Better results without activation units compared to ReLu and ELU 


## Dilated convolutions

Dilated convolution increased receptive field without adding parameters.
Alternative to increasing kernel size.
3x3 kernel with dilation rate 2 = 7x7 receptive field, dilation rate 3 = 11x11 receptive field

DRN  —  Dilated Residual Networks
https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5
! shows equations clearly, good pictures
DRN outperforms ResNet for same parameter counts

Can give gridding artifacts, when high-frequency content present in input


## Grouped convolutions
Aka Filter groups

Partitions inputs into mutually exclusive groups.
Reduces number of computations.

Computational cost.
O output features
R input features
G number of groups
Standard convolution of R×O.
Group convolution to R×O/G


Alexnet variation used multiple groups. Concatenated at the end.
Primarily to allow training in parallel
With 2 groups got better score, at fewer parameters.
With N groups, each convolutional layer is 1/N as deep
https://blog.yani.io/filter-group-tutorial/

Not supported by X-CUBE-AI.

ResNeXt
Adds grouped convolutions to ResNet.
Keras implementation of ResNeXt
https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

ShuffleNet use grouped convolutions inside in each block
ShuttleNet adds a channel shuffle to intermix information in different channels

ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
https://arxiv.org/abs/1807.11164
Half of the features in block are passed on as-is. Similar to DenseNet
has some theoretical experiments on modern CNNs for optimizing efficiency
G1) Equal channel width minimizes memory access cost (MAC)
G2) Excessive group convolution increases MAC
G4) Element-wise operations are non-negligible.
Element-wise operators include ReLU, AddTensor, AddBias, etc.
They have small FLOPs but relatively heavy MAC

For example, at 500MFLOPs ShuffleNet v2 is 58% faster than MobileNetv2,
63% faster than ShuffleNet v1 and 25% faster than Xception.
On ARM, the speeds of ShuffleNet v1, Xception and ShuffleNet v2 are comparable; however,
MobileNet v2 is much slower, especially on smaller FLOPs.
We believe this is because MobileNet v2 has higher MAC


Has this been applied to audio/spectrograms?
Could one do filter groupings along frequency axis?
Could one also perform groupings with multi-scale inputs?

Smart application could also reduce RAM usage?


## Global Average Pooling

Can be used instead of Flatten, to reduced number of channels before Dense
Or can be used to replace Dense layers completely.

Striving for Simplicity: The All Convolutional Net
https://arxiv.org/abs/1412.6806

Replaces Dense layers with Global average pooling
Replaces Maxpooling with Conv with striding


Network in Network (2013)
https://arxiv.org/abs/1312.4400
Review http://teleported.in/posts/network-in-network/

Used Global Average Pooling insted of Dense layers
Last Conv layer has n_classes channels.
Fewer parameters. Improved spatial invariance. More intuitive, less black-box.

DenseNet and ResNet use same

Densely Connected Convolutional Networks
https://arxiv.org/abs/1608.06993

Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385

GAP used a lot for segmentation


## Automatic model search


Learning Transferable Architectures for Scalable Image Recognition
https://arxiv.org/abs/1707.07012
Introduced NASNet. 
1.2% better in top-1 ImageNet accuracy than the best human-invented architectures
while having 9 billion fewer FLOPS - 28% lower computational demand than state-of-the-art.


Auto-Keras: Efficient Neural Architecture Search with Network Morphism. Haifeng Jin, Qingquan Song, and Xia Hu. arXiv:1806.10282.

https://autokeras.com/

- Random search
- Grid search
- Greedy search
- Baysian optimization

Has pluggable interface for the searching mechanism. Searcher
Has a CNN generator clas.

https://github.com/CiscoAI/amla
NAC/EnvelopeNets
ENAS
DARTS

Single Shot Neural Architecture Search Via Direct Sparse Optimization
https://openreview.net/forum?id=ryxjH3R5KQ
Optimizes from a single full network, instead of trying many different models

DARTS: Differentiable Architecture Search
https://paperswithcode.com/paper/darts-differentiable-architecture-search


https://paperswithcode.com/paper/learning-transferable-architectures-for

Rethinking the Value of Network Pruning
https://paperswithcode.com/paper/rethinking-the-value-of-network-pruning
Channel-wise pruning as network architecture search for CNNs
Re-training pruned network from scratch usually better than reusing full network, contrary to exiting thinking



# Background


However, Environmental Noise Directive defines *indicators* for noise pollution:

$L_{den}$: Designed to assess overall annoyance.
It refers to an annual average day, evening and night period of exposure.
Evening are weighted 5 dB(A) and a night weighting of 10 dB(A).
Indicator level: 55dB(A).

$L_{night}$: Designed to assess sleep disturbance.
It refers to an annual average night period of exposure.
Indicator level: 50dB(A).


Why sound/hearing is important
https://www.hearinglink.org/your-hearing/about-hearing/why-do-we-need-to-hear/


## Acoustic Ecology

Soundscape as social construct. Jøran Rudi, NOTAM
Defines Soundscape and Acoustic Ecology.
Ref Murray Schaeffer
Ref Soundscape Journal

> The attention that noise demands is neither lost on straight-pipe motorcyclists in general.
> So noise is a tool with a large social impact. An analysis of urban noise must not miss that point.

> Soundwalks is yet another genre in soundscape art – basically the listeners are led along a predefined
path of sorts, planned because of its signature sounds and interesting audible events.

An Introduction To Acoustic Ecology
https://naisa.ca/radio-art-companion/an-introduction-to-acoustic-ecology/

Hi-fi/low-fi. The quality of the sound environment. Better with Less masking, more frequeny-temporal variations.
Acoustic horizon. How 'far' one can hear 

Five Village Soundscapes (Schafer, 1978b)
European Sound Diary (Schafer, 1977b)
Schafer’s The Tuning of the World (1977a)

particularly regarded by a community and its visitors are called “soundmarks”–in analogy to landmarks.
Natural examples of the latter include geysers, waterfalls, and wind traps while cultural examples
include distinctive bells and the sounds of traditional activities. 


a hi-fi soundscape can be characterised by its lack of masking from noise and other sounds, with the result that all sounds–of all frequencies–”can be heard distinctly” (Schafer, 43)
He defines a hi-fi soundscape as an environment where “sounds overlap less frequently; there is more perspective-foreground and background” (1977a, 43).

Krause (1993) suggested an equilibrium is also apparent across the audio spectrum.
Acoustical spectrographic maps transcribed from 2,500 hours of recordings confirmed his suspicions:
animal and insect vocalisations tended to occupy small bands of frequencies leaving “spectral niches” (bands of little or no energy)
into which the vocalisations (fundamental and formants) of other animals, birds or insects can fit.

Another characteristic of the pre-industrial revolution, hi-fi soundscape, is that the “acoustic horizon” may extend for many miles.
Thus sounds emanating from a listener’s own community may be heard at a considerable distance, reinforcing a sense of space and position and maintaining a relationship with home.
This sense is further strengthened when it is possible to hear sounds emanating from adjacent settlements, establishing and maintaining relationships between local communities

In the lo-fi soundscape, meaningful sounds (and any associated acoustic colouration), can be masked to such an extent that an individual’s “aural space” is reduced.
Where the effect is so pronounced that an individual can no longer hear the reflected sounds of his/her own movement or speech, aural space has effectively shrunk to enclose the individual, isolating the listener from the environment.

If the masking of reflected and direct sounds is so severe that an individual cannot hear his/her own footsteps–which is common on the streets of many cities–”
one’s aural space is reduced to less than that of human proportions” (Truax 1984, 20).
Under such extreme conditions, sound is either smothered (in the sense that particular sounds are not heard)–or sounds merge and sonic information mutates into anti-information: “noise.”

In the developed world, sound has less significance and the opportunity to experience “natural” sounds decreases with each generation due to the destruction of natural habitats.
Sound becomes something that the individual tries to block, rather than to hear; the lo-fi, low information soundscape has nothing to offer.
As a result, many individuals try to shut it out through the use of double glazing or with **acoustic perfume–music**.
*Music*–the virtual soundscape–is, in this context, used as a *means to control the sonic environment* rather than as a natural expression of it.

## Audio Coding for Environmental Sound Classification

https://www.mdpi.com/1424-8220/17/12/2758/htm#B17-sensors-17-02758
logarimic compression, quantization, differential encoding, Huffman compression
Bitrates between 400-1400 bit/s for 8 frames per second.

! only tested on relatively weak classifiers.
With (40 band and 85 frames/s) mel-spectrograms RandomForest/SVM performed 68/69% accuracy on Urbansound8k.
Down to 65% with 30 band and 8 frames/s.
With 1/3 octave, SVM/RF performed 63% accuracy on Urbansound8k.

Same classification down to 5 bits/word.

## Audio Embeddings for Environmental Sound Classification

* VGGish. 128 dimensions, 8 quantized bit. Used as base for DCASE2019 challenge.
* SoundNet.
* L^3 (Look, Listen, Learn). 73.65% on Urbansound8k. 512 dimensions.
* EdgeL3. Compressed version of L^3.
70% compression. 72.64% on Urbansound8k.
0.814 MB for its 213,491 parameters in float32.
12 MB RAM needed for activations.

## Sensors

### Detection of Anomalous Noise Events on Low-Capacity Acoustic Nodes for Dynamic Road Traffic Noise Mapping within an Hybrid WASN
https://www.mdpi.com/1424-8220/18/4/1272/htm
April 2018 

Hybrid architecture, low-capacity and high-capacity nodes
Anomalous Noise Event Detector (ANED) to run on low-capacity nodes, on microcontroller

Experiments with considering 9 h and 8 min of real-life acoustic data from both urban and suburban environments,
show the feasibility of the proposal both in terms of computational load and in classification accuracy.

### Wireless Sensor Networks for Long-Term Monitoring of Urban Noise
Sensors. September 2018
Courtney Peckens, Cédric Porter and Taylor Rink

A-weigthing filter in hardware
Teensy microcontroller.
Calculating Leq_T.
Xbee tranciever

Mel-filters
Probability Density Functions
Linear discriminator

### Design aspects of acoustic sensor networks for environmental noise monitoring
September 2016.
https://www.sciencedirect.com/science/article/pii/S0003682X16300627

Categorizes ESC sensors into 4 categories,
based on Hardware Costs, Scalability, Flexibility, Accuracy.

Evaluated different usecases.
Noise annoyance, Illegal firework detection/localization, Noise reason monitoring for tram passings. 

### Environmental noise monitoring using source classification in sensors
https://www.sciencedirect.com/science/article/pii/S0003682X17307533
2017
Panu Maijalaa Zhao Shuyang ToniHeittola Tuomas Virtanen
Raspberry PI.
MFCC with GMM, ANN
"In environmental sound classification, a training set with a few hours can currently be regarded as a large dataset"

### Sound monitoring networks new style
2011.
1/3 octave bands.
"Temporal contrast filters"
Self-Organizing Maps


## Machine Learning for hearing aids

SoundSense Learn.
A-B feedback for training automatic settings for hearing aids 
http://www.hearingreview.com/2018/05/real-life-applications-machine-learning-hearing-aids-2/

## Feature representation

[An Optimized Recurrent Unit for Ultra-Low-Power Keyword Spotting](https://arxiv.org/abs/1902.05026).
??? do they use standard Urbansound8k folds and test-set
Uses samplerate 8kHz.
eGRU_arch Urbansound8k score of 72%. Maybe 8kHz is enough?
eGRU_opt UrbanSound8k 61.2%

## WHO Noise Guidelines

The definition of “community noise” used in the CNG in 1999 was also adapted. The GDG agreed to
use the term “environmental noise” instead, and offered an operational definition of: “noise emitted
from all sources except sources of occupational noise exposure in workplaces”.

## UM2526 - Getting started with X-CUBE-AI Expansion Package for Artificial Intelligence (AI)

STM32 Arm ® Cortex ® -M4: ~9 cycles/MACC

add-on “AI System Performance”

6 X-CUBE-AI internals

> The C-code generator optimizer engine seeks to optimize memory usage (RAM & ROM) against inference computing time (power consumption is also considered).

> Optimal activation/working memory:
> A R/W chunk is defined to store temporary hidden layer values (outputs of the activation operators).
> It can be considered as a scratch buffer used by the inference function. 
> The activation memory is reused across different layers. As a result, the activation buffer size is defined by the maximum memory requirements of two consecutive layers.


# Quantization

Incremental network quantization: Towards lossless CNNs with low-precision weights. April 2017.
https://arxiv.org/abs/1702.03044
No accuracy drop on ResNet-50 for 4,3 and 2-bit weights.
Gives power-of-two weights. Can be performed with bit-shifts, no multiply needed.

LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks.
Jul 2018.
http://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html
https://github.com/Microsoft/LQ-Nets - implemented for TensorFlow 1.3+
Can quantize to bitwise operations for fast inference.
Down to 2 bit weights and activations. Very close performance to full precision.
Does not quantizize first and last layer, claims speedup from bitwise operations low due to few channels.
Training time is 1.4x to 3.8x longer than without quantization (depending on quant setting).
! large list of references on quantized networks.

Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
http://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html
Up to 200% faster inference on ARM smartphones.
Notes that ReLu6 benefical, since it creates a natural (0, 6) range for activations.

Training and Inference with Integers in Deep Neural Networks. February 2018.
https://arxiv.org/abs/1802.04680
2-bit Ternary weights. No multiplications during inference.
8-bit actications.
Proposes WAGE. Constraints on Weights, Activations, Gradients and Errors to low-bithwidth integers,
*both in training and inference*.
Introduce a new initalization method and scaling layer to replace Batch Norm, which can cause problems for quantization.
Table 5: energy costs in silicon. 16-bit FP and 32-bit FP use 5-10x and 15-30x more energy than 8-bit INT.
"It is promising to training DNNs with integers encoded with logarithmic representation"

ShiftCNN: Generalized Low-Precision Architecture for Inference of Convolutional Neural Networks
https://arxiv.org/abs/1706.02393
Power-of-two weight representation and, only shift and addition operations. Multiplierless CNN.
Also precomputing convolution terms. Can be applied to any CNN architecture with a relatively small codebook of weights, allows to decrease the number of product operations by at least two orders of magnitude
Quantize full-precision floating-point model into ShiftCNN *without retraining*.
? can it be implemented efficiently in software, without custom ShiftALU hardware?

SEP-Nets: Small and Effective Pattern Networks
https://arxiv.org/abs/1706.03912
Binarizes kxk convolutions, 1x1 convolutions left as is or 8-bit quantizied
Also uses 4x grouped convolutions.
Near Mobilenet performance on Imagenet at 1/5 the size, 1MB 

Ristretto
http://ristretto.lepsucd.com/
Operates on Caffee models.
Supports Convolutional and Fully Connected layers.
? no support for Depthwise Separable layers?
! Not maintained since September 2016.
! Dynamic-fixed-point seems not executable as "ordinary fixed point" without hacks?
With `ristretto` CLI tool can specify desired error rate, and will find smallest bitwidths for parameters. 
Example on SqueezeNet less than 1% degradation in performance.
Supports several quantization strategies.
* Dynamic Fixed Point: A modified fixed-point format.
* Power-of-two parameters: Layers with power-of-two parameters don’t need any multipliers, when implemented in hardware


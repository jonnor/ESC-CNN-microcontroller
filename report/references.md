
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
83.7% on UrbanSound8k.
Uses mixup and data augmentation. 5% increase in perf
Uses stacked 1-D convolutions in some places
(3,7), (3,5), (3,1), (3,1), (1,5), 

log-melspectrogram, gammatone compared
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

81.9% accuracy on Urbansound8k
ReLu 81.2%

log-mel + delta log-mel
Time stretching, noise additon

### Environmental sound classification with dilated convolutions
https://www.sciencedirect.com/science/article/pii/S0003682X18306121
December, 2018

DilaConv

Claim: Dilated CNN achieves better results than that of CNN with max-pooling.
About 4% on UrbanSound8K.
! not compared with striding
! no mention of data augmentation applied??

log-mel + delta-logmel

sample rate 22500
window size of 1024, hop length of 512
64 n-mels
41 frames
Global Average Pooling, Softmax output
SGD 0.0001, 0.9 Nesterov momentum
500 epochs training

Dilated convolution increased receptive field without adding parameters.
3x3 kernel with dilation rate 2 = 7x7 receptive field, dilation rate 3 = 11x11 receptive field
! great images in the article.

### LD-CNN: A Lightweight Dilated Convolutional Neural Network for Environmental Sound Classification
2018.
http://epubs.surrey.ac.uk/849351/1/LD-CNN.pdf

LD-CNN
Based on D-CNN, but 50 times smaller, only looses 2% points accuracy

log-mel + delta-logmel
time-stretching augmentation
60 mels

31 frames on Urbasound
101 frames on ESC50

SDG 0.01 learning rate, 0.9 momentum
mean voting between windows

Early layers use 1D stacked convolutions.

Model size 2.05MB.
79% Urbansound.
66% ESC-50.

References multiple other lightweight ESC models.
?? Claims DenseNet performs well at 390.3KB size. But their reference does not support this.


### WSN: COMPACT AND EFFICIENT NETWORKS WITH WEIGHT SAMPLING
https://arxiv.org/abs/1711.10067
Xiaojie Jin, Yingzhen Yang

2018.

CNN trained on raw audio.
8 layers

Compared on UrbanSound8k and ESC-50.
70.5% average acc on UrbandSound8k.
! evaluated on 5 folds? the dataset is pre-stratified with 10 folds, leakage can happen



2 model variations evaluated, plus quantized versions.
520K and 288K parameters.


### Convolutional Neural Networks for Small-footprint Keyword Spotting 
2015
https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

Designs small CNNs
Keeping number of multiples the same, at 500 M
Number of params around 50-100k

40 log-mel
25 ms frames
10 ms shift
32 frame windows = 310 ms
! 1 frame shifts = 96% overlap

1 layer or 2 layer CNNs. Testing pooling and striding alternatives

Getting very good results for striding in time and striding in frequency


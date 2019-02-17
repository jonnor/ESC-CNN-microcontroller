# Datasets

Audio Classification datasets that are useful for practical tasks
that can be perform on microcontrollers and small embedded systems.

Relevant tasks:

* Wakeword detection
* Keyword spotting
* Speech Command Recognition
* Noise source identification
* Smart home event detection. Firealarm,babycry etc 

Not so relevant:

* (general) Automatic Speech Recognition
* Speaker recognition/identification

## To be reviewed.

* DCASE 2016.

## Not relevant 

* [NOIZEUS: A noisy speech corpus for evaluation of speech enhancement algorithms](http://ecs.utdallas.edu/loizou/speech/noizeus/)
30 sentences corrupted by 8 real-world noises. 
* [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/), 100k utterances for 1251 celebrities.
Task: Speaker Reconition.
* [Speakers in the Wild](https://www.sri.com/work/publications/speakers-wild-sitw-speaker-recognition-database)
Task: Speaker Reconition.
* [Google AudioSet](https://research.google.com/audioset/).
2,084,320 human-labeled 10-second sounds, 632 audio event classes. 
Based on YouTube videos.
* Whale Detection Challenge. https://www.kaggle.com/c/whale-detection-challenge
* Mozilla [Common Voice](https://voice.mozilla.org), crowd sourcing.
Compiled dataset [on Kaggle](https://www.kaggle.com/mozillaorg/common-voice), 
500 hours of transcribed sentences.
Has speaker demographics.
Task: Automatic Speech Recognition.
Not something to do on microcontroller.
Could maybe be used for Transfer Learning for more relevant speech tasks.
* DCASE2018 Task 5.
Domestic activities. 
10 second segments. 9 classes.
From 4 separate microphone arrays (in single room).
Each array has 4 microphones

## Relevant but lacking

* Hey Snips. https://github.com/snipsco/keyword-spotting-research-datasets
Task: Wakeword detetion, Vocal Activity Detection.
Restricted licencese terms. Academic/research use only.
Must contact via email for download.
By Snips, developing Private-by-Design, decentralized, open source voice assistants.

## Relevant

### TUT Rare Sound Events 2017

Used for DCASE2017 Task 2.
Baby crying, Glass Breaking, Gunshot.
3 classes, but separate binary classifiers encouraged.
Part of TUT Acoustic Scenes 2016.
Train 100 hours. Approx 100 sound examples per class isolated, 500 mixtures (weakly labeled).
Event detection.  Event-based error rate. Onset only. 500 ms collar. Also F1 score.
Baseline system available. FC DNN. 40 bands melspec, 5 frames. F1 0.72
Around 11 other systems submitted. Ranging 0.65-0.93 F1 score.
http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-rare-sound-event-detection

Relevant as examples of single-function systems, security

### TensorFlow Speech Commands Data Set
Task: Keyword spotting / speech command


DSCNN-L  [How to Achieve High-Accuracy Keyword Spotting on Cortex-M Processors](https://community.arm.com/processors/b/blog/posts/high-accuracy-keyword-spotting-on-cortex-m-processors)

Very well explored.

Has state-of-the-art results for microcontroller.
https://arxiv.org/abs/1711.07128
Running on STM32F746G-DISCO. DSCNNL model gives 0.83/84.6% on Kaggle leaderboard
[How to Achieve High-Accuracy Keyword Spotting on Cortex-M Processors](https://community.arm.com/processors/b/blog/posts/high-accuracy-keyword-spotting-on-cortex-m-processors).
Reviews many deep learning approaches. DNN, CNN, RNN, CRNN, DS-CNN.
Considering 3 different sizes of networks, bound by NN memory limit and ops/second limits.
Small= 80KB, 6M ops/inference.
Depthwise Separable Convolutional Neural Network (DS-CNN) provides the best accuracy while requiring significantly lower memory and compute resources.
94.5% accuracy for small network.
8-bit weights and 8-bit activations, with KWS running at 10 inferences per second.
Each inference – including memory copying, MFCC feature extraction and DNN execution – takes about 12 ms.
10x inferences/second. Rest sleeping = 12% duty cycle.
!uses MFCC instead of log-melspectrogram. melspec usually performs better?
?what is the compute time balance between MFCC feature calculation and inference?

https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data

20 core words. Said multiple times per speaker.
"Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine".
10 auxiliary words. 1 repetition per speaker.
"Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", and "Wow".
Note. Only 12 possible labels for the Test set: yes, no, up, down, left, right, on, off, stop, go, silence, unknown.
Some background noise files are also available.
39 participants scored 0.90-0.91060 (averaged Multiclass Accuracy).
Has 64,727 audio files total.
Lengths trimmed to be around 1 second. Aligned by the loudest utterance.
Released on August 3rd 2017.
Licensed Creative Commons 4.0.
A Special Prize required submissions to:
Have model size below 5MB.
Run in below 200ms on a Raspberry PI3.
They provided a benchmark script for evaluating it.
Winner Heng-Ryan-See * good bug? got 0.90825, very near top of any model.
Explanation of solution, with tips & tricks.
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46945
Used pseudo-labelling to generate "silence" and "unknown" data.
Single model at 0.87-0.88 without augmentation. Using ensembles to get higher.
Says melspectrogram perform better than MFCC.
A simple CNN reaching 0.74. https://www.kaggle.com/alphasis/light-weight-cnn-lb-0-74
A 1D CNN reaching 0.77. https://www.kaggle.com/kcs93023/keras-sequential-conv1d-model-classification
Has a tutorial at https://www.tensorflow.org/tutorials/sequences/audio_recognition
! uses MFCC as input features.
CNN architecture there is based on www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
Example code for running in streaming mode is also provided (C++).
Has several architecture alternatives.
Ex `low_latency_svdf` with 750K parameters and 750kFLOPS. Reaching 0.85 acc
based Compressing Deep Neural Networks using a Rank-Constrained Topology, https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf
`tiny_conv` designed for microcontrollers, 20KB RAM and 32KB FLASH.

Studying the Effects of Feature Extraction Settings on the Accuracy and Memory Requirements of Neural Networks for Keyword Spotting, https://ieeexplore.ieee.org/abstract/document/8576243
Compares MFCC feature extraction settings. 10-40 MFCC bands. 10-40ms size. Compares RAM/ROM use.
! GRU used 0.66 KB RAM with 314KB FLASH reaching 92%.
DS-CNN used 62KB RAM with 161KB FLASH reaching 93.5%

Novel architecture explored
[On-the-fly deterministic binary filters for memory efficient keyword spotting applications on embedded devices](https://dl.acm.org/citation.cfm?id=3212731)
BinaryCmd makes represents weights as a combination of predefined orthogonal binary basis.
that can generate convolutional filters very efficiently on-the-fly.
Deter-ministic Binary Filters, DBF.
Orthogonal variable spreading factor, OVSF
Also using MFCC features.
Claims state of the art results (over Hey Edge) for models under 3MOPs and 30kB of model size.
! but CRNN at 3.3MOP has much higher performance
! smaller variation of CRNN nor DS-CNN not tested...

FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network
https://www.microsoft.com/en-us/research/publication/fastgrnn-a-fast-accurate-stable-and-tiny-kilobyte-sized-gated-recurrent-neural-network/
Evaluated on Google Speech Command Set, both 30 and 12 class. Clips are 1 second
12 class: Smallest model 5.5KB, 92% acc, 242 ms on Cortex M0+ @ 48Mhz.
Using Log mel spectrograms, 32 mels, 25ms window, 10ms stride



Veniat2018StochasticAN
Keyword spotting.
Designing multiple architectures with different complexity,
switching automatically at runtime to use simpler models to reduce CPU time 
Evaluated on Speech Command Set and http://github.com/TomVeniat/SANAS
cnn-trad-fpool3 used 120-130 MFLOPS/frame for 72.8% correct,
their solution 40M for 80% correct and higher match for matched or slightly better perf.

### ESC-50

[ESC-50: Dataset for Environmental Sound Classification](https://github.com/karoldvl/ESC-50).
2k samples, 50 classes in 5 major categories.
5 seconds each.
Compiled from freesound.org data
! only 40 samples per class.

Relevant for context-aware-computing, smarthome, environmental noise source prediction?

Github repo has an excellent overview of attempted methods and their results.

* Best models achieving 86.50% accuracy.
* Human accuracy estimated 81.30%.
* Baseline CNN at 64.50%. 
* Baseline MFCC-RF, 44.30%.
* Over 20 CNN variations attempted.

What resource-efficient methods exist?

How would FastGRNN do on this dataset?
How would FastGRNN do in combination with learned audio->2d features?

#### AclNet: efficient end-to-end audio classification CNN
https://arxiv.org/abs/1811.06669
November, 2018

Depthwise-Separable @44kHz. 81.75% accuracy, 155k parameters, 49M multiply-adds/second. 
Standard Convolutio @44kHz, 82.20% accuracy, 84k parameters, 131 multiply-adds/second.
16kHz topped at 80.9%, just below human perf.
! 16kHz Standard Convolution not tested?
Using data augmentation and mixup. 5% improvement. a=0.1/0.2 for mixup
Using two layers of 1D strided convolution as FIR decimation filterbank.
Then a VGG type architecture.

Is this strided convolution on raw audio
more computationally efficent than STFT,log-mel calculation?
LLF 1.44k params, 4.35 MMACS. 2 conv, maxpool.
1.28 second window. 64x128 output. Equivalent to 64 bin, 10ms skip log-mel?
Can it be performed with quantizied weights? 8 bit integer. SIMD.
Would be advantage, FFT is hard to compute in this manner..
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

### WaveMsNet, Learning Environmental Sounds with Multi-scale Convolutional Neural Network
https://arxiv.org/abs/1803.10219
March, 2018

When only using raw audio. 
When combined with log-mel features, 79.10% on ESC-50. 93.75% on ESC-10.

Uses 1-d convolutions on raw audio in frontend.
Tests small,medium,large receptive field, and a model with all.
Multi-scale outperforms single by 2-3% absolute.


### Learning environmental sounds with end-to-end convolutional neural network
EnvNet

https://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK.pdf
When combined with log-mel, gives 6.5% improvement.


Learned CNN frontend outputs 40x150 feature map.
Tests different filter sizes for low level features. For 44.1kHz.
Find 8 long performs best. With 2 conv layers.
! graphs showing frequency response of the learned layers.
When filter indexes are sorted, has similar shape to mel-scale.

On ESC-50. Input lengths of 1-2.5 seconds perform similarly.

logmel only. 58.9%
logmel,logmel-delta. 66.5%

EnvNet raw. 64%
EnvNet logmel,raw. 69.3%
EnvNet logmel,logmeldelta,raw. 71%

Uses a 1 second window. Randomly chosen at training time.
Longer clips are classified by probabalistic voting over windows. 0.2second stride 

### Learning from Between-class Examples for Deep Sound Recognition
https://openreview.net/forum?id=B1Gi6LeRZ
Feb 2018.

EnvNet-v2 is like EnvNet but with
44.1 kHz instead of 16kHz,
13 layers instead of 7.

Without special learning, 69.1% on Urbansound8k
Using between-class learning and strong augmentation, got
84.9% on ESC-50, 91.4% on ESC-10, 78.3% on Urbansound8k 

### Very deep convolutional neural networks for raw waveforms

Evaluated on UrbanSound8k.
Uses 8kHz sample rate.

ResNet type architecture.
Testing 3-34 layers.
M=18 layers performed best. 71%. 3.7M parameters
M5-big 63.30%, 2.2M parameters.


Reproduction and Keras implemention
https://github.com/philipperemy/very-deep-convnets-raw-waveforms
! "Going really deep does not seem to help much on this dataset. We clearly overfit very easily"

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
M 2048 75 44.1 top1=74.70% 88.35%

75% overlap better than 50%.
.. Down to 66% with 32/16kHz sample rate.
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
Urbansound8k 73.6%. ESC-50 71.1±0.8  ESC-10 84.1±0.7

Fusion with 3 heads performs better.



### Look, Listen and Learn
August, 2017.

Unsupervised learning of audio+image embeddings.
Using log-spectrogram input.
Output audio: 512 dimensional vector.

ESC-50. 79.3%

## Urbansound-8k

[Urbansound-8k](https://urbansounddataset.weebly.com/urbansound8k.html)

8000 samples total. 10 classes.
Compiled from freesound.org data.
Relevant for environmental noise source prediction.

! recommendation. Use the predefined 10 folds and perform 10-fold (not 5-fold) cross validation.
Otherwise will get inflated scores, due to related samples being mixed.

### UNSUPERVISED FEATURE LEARNING FOR URBAN SOUND CLASSIFICATION
2015.

Using spherical-k-means to learn single layer of convolutions.

Reaches 72% accuracy.
log-melspectrogram input.
Patches were 128 band tall. 8 frames long.
k=2000
128*8*2000 = 2.1M parameters

### Deep Convolutional Neural Network with Mixup for Environmental Sound Classification
https://link.springer.com/chapter/10.1007/978-3-030-03335-4_31
November, 2018.
83.7% on UrbanSound8k.
Uses mixup and data augmentation. 5% increase in perf
1-D convolutions in some places instead of 3x3.

### Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
Justin Salamon and Juan Pablo Bello.
November 2016.
https://arxiv.org/pdf/1608.04363.pdf
SB-CNN. 73% without augmentation, 79% with data augmentation.
3-layer convolutional, using 5x5 conv and max pooling.
References models PiczakCNN 72% avg acc and SKM 72% avg acc. 
? Baseline candidate.
Parameters not specified. Estimated 444k

### ENVIRONMENTAL SOUND CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORKS
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

### LEARNING FILTER BANKS USING DEEP LEARNING FOR ACOUSTIC SIGNALS. Shuhui Qu.
Based on the procedure of log Mel-filter banks, we design a filter bank learning layer.
Urbansound8K dataset, the experience guided learning leads to a 2% accuracy improvement.

### WSN: COMPACT AND EFFICIENT NETWORKS WITH WEIGHT SAMPLING
2018.
CNN trained on raw audio.
Compared on UrbanSound8k and ESC-50.
70.5% average acc on UrbandSound8k.
! evaluated on 5 folds? the dataset is pre-stratified with 10 folds, leakage can happen
2 model variations evaluated, plus quantized versions.
520K and 288K parameters.
1.0e9 mult-adds.
SoundNet used as baseline.

SoundNet.
Transfer learning

### Listening to the World Improves Speech Command Recognition
2017.

Uses multi-resolution approach with dilated convolutions.
4 different dilations.
Using padding to keep them the same size, and stacking along channel dimension.
Uses Urbansound8k -> Google Speech Command transfer learning.

### Environmental sound classification with dilated convolutions
https://www.sciencedirect.com/science/article/pii/S0003682X18306121
December, 2018
Dilated CNN achieves better results than that of CNN with max-pooling.
About 4% on UrbanSound8K.
! not compared with striding

Dilated convolution increased receptive field without adding parameters.
3x3 kernel with dilation rate 2 = 7x7 receptive field, dilation rate 3 = 11x11 receptive field
! great images in the article.

### Audio Event Classification using Deep Learning in an End-to-End Approach
Jose Luis Diez Antich
Explored end-2-end learning, using raw audio as input.
Was unable to reach more than 62% average accuracy.
? how many parameters

### LD-CNN: A Lightweight Dilated Convolutional Neural Network for Environmental Sound Classification
2018.
http://web.pkusz.edu.cn/adsp/files/2015/10/ICPR_final.pdf

Early layers use 1D stacked convolutions.

Model size 2.05MB.
79% Urbansound.
66% ESC-50.

References multiple other lightweight ESC models.
?? Claims DenseNet performs well at 390.3KB size. But their reference does not support this.


## YorNoise
Dataset from York with "traffic" and "rail" classes.
Same structure as Urbansoun8k dataset.
1527 samples a 4 seconds. Split into 10 folds.
https://github.com/fadymedhat/YorNoise

## DCASE2018 Task 2, General Purpose Audio Tagging
Task: Acoustic event tagging.
Based on FreeSound data.
41 classes. Using AudioNet ontology
9.5k samples train, ~3.7k manually-verified annotations and ~5.8k non-verified annotations. 
Test  ~1.6k manually-verified annotations.
Systems reach 0.90-0.95 mAP@3.
Baseline CNN on log melspec. 0.70 mAP@3

Relevant for: context-aware-computing, smarthome?

## DCASE2018 Task3, Bird Audio Detection.
Binary classification.

Relevant for on-edge pre-processing / efficient data collection.

## DCASE2018 Task4
Event Detection with precise time information.
Events from domestic tasks. 10 classes.
Subset of Audioset.

Relevant for: smarthome and context-aware-computing


## TUT Urban Acoustic Scenes 2018
Used in DCASE2018 Task 1.

Task: Acoustic Scene Classification.
10 classes. airport,shopping_mall,metro_station
About 30GB of data, around 24 hours training.
One variant dataset has parallel recording with multiple devices, for testing mismatched case.

Relevant for: context-aware-computing?

## TUT Acoustic Scenes 2017. Used for DCASE2017 Task 1.
Scenes from urban environments.
15 classes.
10 second segments.
Baseline system available. 18% F1
Relatively hard, systems achieved 35-55% F1.

Relevant for context-aware-computing?

## DCASE2017 Task 4, Large Scale Sound Event detection
http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-large-scale-sound-event-detection
17 classes from 2 categories, Warning sounds and Vehicle sounds.

Relevant for autonomous vehicles?

## TUT Sound Events 2017
Used for DCASE2017 Task 3, Sound event detection in real life audio

Events related to car/driving.
6 classes.
Multiple overlapping events present. Both in training and testing.
Hard, systems only achieved 40%-45% F1.
Quite small. 2 GB dataset total.
Relevant for autonomous vehicles?



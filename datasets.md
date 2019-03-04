

### TensorFlow Speech Commands Data Set
Task: Keyword spotting / speech command


DSCNN-L [How to Achieve High-Accuracy Keyword Spotting on Cortex-M Processors](https://community.arm.com/processors/b/blog/posts/high-accuracy-keyword-spotting-on-cortex-m-processors)

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


#### AclNet: efficient end-to-end audio classification CNN
https://arxiv.org/abs/1811.06669
November, 2018

ESC-50
Depthwise-Separable @44kHz. 81.75% accuracy, 155k parameters, 49M multiply-adds/second. 
Standard Convolutio @44kHz, 82.20% accuracy, 84k parameters, 131M multiply-adds/second.
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

## Urbansound-8k

[Urbansound-8k](https://urbansounddataset.weebly.com/urbansound8k.html)

8000 samples total. 10 classes.
Compiled from freesound.org data.
Relevant for environmental noise source prediction.

! recommendation. Use the predefined 10 folds and perform 10-fold (not 5-fold) cross validation.
Otherwise will get inflated scores, due to related samples being mixed.




### LEARNING FILTER BANKS USING DEEP LEARNING FOR ACOUSTIC SIGNALS
Shuhui Qu.

Based on the procedure of log Mel-filter banks, we design a filter bank learning layer.
Urbansound8K dataset, the experience guided learning leads to a 2% accuracy improvement.






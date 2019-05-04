
<!---

Contributions

- Demonstrate an ESC system running on a 5 USD microcontroller.
Resulting in XX% accuracy on Urbansound8k dataset.
- Set of tools for STM32 CubeMX AI
- 

-->


\newpage
# Introduction

<!---


Noise assessment
Challenges.
    Local problem, multiple sources, time-dependent
    Perceptual/subjective evaluation
    Positive sound qualities. Recreational
Increasing noise problem. Urbanization

Track noise level, plan/take corrective action. Visibility
Identifying source. Environmental Sound Classification
Smart-city concept, data-driven

Privacy, GDPR
=> this thesis

-->


<!--
## Measuring noise
Noise level is measured as Sound Pressure Level (SPL), most commonly expressed in decibel.

over the reference of 20*10-6 Pa

Sound levels change 
Environmental noise is measured using continious equivalent sound level ($L_{eq_T}$) over
a some time-period T.
A common time-period used for health is 1 year. 
The equivalent sound level is the sound level that which has the same
energy


It is common to distinguish between noise that occurs in the daytime ($L_{day}$),
and at nighttime ($L_{night}$). 
The yearly 

Sleepers that are exposed to night noise levels above 40dB on average throughout
the year can suffer health effects like sleep disturbance and awakenings.
Above 55dB long-term average exposure, noise can trigger elevated blood pressure and lead to ischemic heart disease.
The WHO has set a Night Noise Guideline level for Europe at 40 dB $L_{night}$.

-->



## Environmental noise

Noise is a growing problem in urban areas, and due to increasing urbanization more and more people are affected.
Major sources of noise include transportation, construction, industry and recreational activities.
The sum of all the noise is referred to as Environmental noise or noise pollution.

Noise pollution over sustained periods of time affects health and well-being in many ways.
Noise can be a source of annoyance and increased stress, cause sleeping disturbance
and increase risk of heart diseases.
WHO has estimated that in Europe 1.6 million healthy life years (Disability-Adjusted Life Years, DALY)
are lost annually due to noise pollution[@WHONoiseBurden2018].
This makes noise pollution the second environmental cause of health problems in Europe, after air pollution.

![Health impacts of noise at different severity levels[@NoiseStressConcept]](./img/noiseseverity.png)

`FIXME: top triangle has poor text contrast.`

In the EU, Environmental noise is regulated by Environmental Noise Directive (2002/49/EC)[@EuNoiseDirective].
The purpose of the directive is to:

* Determine people's exposure to environmental noise
* Ensuring that information on environmental noise and its effects is available to the public
* Preventing and reducing environmental noise where necessary
* Preserving environmental noise quality where it is good

Member States of the EU are required create noise maps and noise management action plans every 5 years.
These must cover all urban areas, major roads, railways and airports over a certain size.

The noise maps are created using simulation of known noise sources (such as car traffic)
with mathematical sound propagation models, based on estimates for traffic numbers.
These maps only give yearly average noise levels.

<!--
TODO: Add paragraph about noise nuisance.
In addition to the health effects of long-term noise source,
inhabitants can also be bothered by short-term. 

TODO: add info about reduced property value

MAYBE: mention occupational noise?
-->

## Noise monitoring with Wireless Sensor Networks

Several cities have started to deploy networks of sound sensors in order to better understand and reduce noise issues.
These sensor networks consist of many sensor nodes positioned in the area of interest,
transmitting the data to a central system for storage and reporting.

Examples of established projects are Dublin City Noise[@DublinCityNoise] with 14 sensors across the city since 2016.
The Sounds of New York City (SONYC)[@SONYC] project had 56 sound sensors installed as of 2018[@SONYC2019],
and the Barcelona Noise Monitoring System[@BarcelonaNoiseMonitoring] had 86 sound sensors[@BarcelonaNoiseMonitoring2018].
Future projects include CENSE[@CENSE], which plans to install around 150 sensors in Lorient, France[@CENSESensor].

![Illustration of how Sounds of New York City[@SONYC-CPS] system combines sensor networks and citizen reporting with data analysis and to present city experts and agencies with a visual interactive dashboard "Noise Mission Control".](./img/SONYC-CPS.png){ width=50% }

To keep costs low and support a dense coverage, the sensor nodes are can be designed to operate wirelessly.
Communication is done using wireless radio technologies such as WiFi, GSM, NB-IoT or 6LoWPAN.
The sensor harvests its energy, commonly using solar power or from streetlights powered at night.
A battery backup allows the sensor to continue operating also when energy is momentarily unavailable.

These sensor networks enable continuous logging of the sound level.
(Leq dB) `TODO: explain Leq dB`
Typical measurement resolution are per-minute, per second or per 125ms.
In Europe sound level sensors are designed to specifications of IEC 61672-1 Sound Level Meters[@IECSoundLevelMeters].
The equivalent standard for North America is ANSI S1.4[@ANSISoundLevelMeters].

Sensors can also provide information that can be used to characterize the noise,
for instance to identify the likely noise sources.
This is desirable in order to understand the cause of noise,
identify which regulations the noise falls under, which actors may be responsible,
and to initiate possible interventions.

This requires much more data than sound level measurements,
making it challenging to transmit the amount of data within the bandwidth and energy budget of a wireless sensor.
The sensor may also capture sensitive information and violate privacy requirements by
recording and storing such detailed data.

To address these concerns several methods for efficiently coding the information before transmitting to the server have been developed.
See Figure \ref{figure:sensornetworks-coding} for an overview.

\begin{figure}[h]
  \centering
    \includegraphics[width=1.0\textwidth]{./img/sensornetworks.png}
\caption{Different data transmission strategies for a noise sensor network with noise source classification capability.
A) Sensor sends raw audio data with classification on server.
B) Sensor sends spectrograms as a intermediate audio representation. Classification on server.
C) Sensor sends neural network audio embeddings as intermediate audio representation. Classification on server.
D) Sensor performs classification on device and sends result to server. No audio or intermediate needs to be transmitted.
}
\label{figure:sensornetworks-coding}
\end{figure}


In [@AudioCodingSensorGrid], authors propose a compressed noise profile based on lossy compression of spectrograms.
For 125ms time resolution the bitrate is between 400 and 1400 bits per second,
however this gave a 5 percentage points reduction in classification accuracy.
This is shown as case B. of Figure {figure:sensornetworks-coding}.

Others have proposed to use neural networks to produce an audio "embedding" inspired
by the success of world embeddings for Natural Language Processing (case C. of Figure {figure:sensornetworks-coding}).
In VGGish[@VGGish] model trained on Audioset[@AudioSet]
a 8-bit, 128 dimensional embedding is used for 10 seconds clips,
leading to a datarate of 102 bits per second.
L^3 (Look, Listen, Learn)[@L3] similarly proposed an embedding with 512 dimensions.
The computation of such an embedding generally requires very large models and lots of compute resources.
EdgeL^3[@EdgeL3] showed that the L^3 model can be compressed by up to 95%,
however the authors state that more work is needed to fit the RAM constraints of desirable sensor hardware.

The minimal amount of data transmissions would be achieved if only sending the detected noise category,
requiring to perform the classification on the sensor, case D. of figure {figure:sensornetworks-coding}.

This motivates the problem statement of this thesis:

> Can we classify environmental sounds directly on a wireless and battery-operated noise sensor?


<!---
Measuring noise

    Terminology. Sound Pressure Level
    Frequency weighting
    Summarizaton. Eq SPL. L10, L90, Lmax, Lpeak
    Spectrograms. Time-frequency characteristics. 1/3 acoustic bands
    Equipment

Sensor Networks for Noise Monitoring

    Research projects.
    Commercially available units.

    Sound sensor. Microphone
    Processing unit. Single-board-computers, microcontrollers
    Connectivity. WiFI, GSM, LoRa. Bandwidth, range, power consumption
    Energy source. Battery and Energy harvesting.
-->

\newpage
# Background

<!---
Related works for our task...
-->

<!---
DROP

Acoustic Scene Classification. Context-dependent compute
Domestic Sound Classification. 
Event detection. Not fine identification in time (onset)
Open-set classification. Novelty detection, clustering
-->


## Digital sound

Physically, sound is a variation in pressure over time.
For machine learning, it must be exist in a digital representation.
The acoustic data is converted to analog electric signals by a microphone and
then digitized using an Analog-to-Digital-Converter (ADC),
as illustrated in Figure \ref{figure:audio-aquisition}.

![Conversion of sound into a digital representation \label{figure:audio-aquisition}](./img/audio-aquisition.png)

In the digitization process, the signal is quantized in time at a certain sampling frequency,
and the amplitude quantized at a certain bit-depth.
A typical sampling frequency is 44100 Hz, and bit-depth 16 bit.
With these parameters, the acoustic sound can be reconstructed without perceivable differences
to the human ear.
In this representation, sound is a 1 dimensional sequence of numbers.
This is sometimes referred to as a *waveform*. 

Digital sound can be stored uncompressed (example format: WAV),
using lossless compression (FLAC)
or using lossy compression (MP3).
Lossy compression removes information that are indistinguishable to the human hear
and can compress better than lossless.
It however adds compression artifacts, and is best avoided for machine learning tasks.

Recordings can have multiple channels of audio but for machine learning on audio
single-channel data (mono-aural) is still the most common.

### Spectrograms

Sounds of interest often have characteristic patterns not just in time (temporal signature)
but also in frequency content (spectral signature).
Therefore it is common to analyze the audio in a time-frequency representation (a *spectrogram*). 

The standard way to transform audio waveform into a spectrogram is by using the Short Time Fourier Transform (STFT).
The STFT operates by splitting the audio up in short consecutive chunks,
and computing the Fast Fourier Transform (FFT) to estimate the frequency content for each chunk.
To reduce artifacts at the boundary of chunks, they are overlapped (typically by 50%)
and a window function (such as the Hann window) is applied before before computing the FFT.

There is a trade-off between frequency (spectral) resolution and time resolution with the STFT.
The longer the FFT window the better the frequency resolution,
but the poorer the temporal resolution.
For speech a typical choice of window length is 25 ms.
Similar frame lengths are often adopted for acoustic events.

`TODO: add references`
`TODO: image of windowing / STFT process`
![Computing a STFT spectrogram from an audio waveform](./images/frames.png)

<!--
Inspiration for TODO: image, https://dsp.stackexchange.com/questions/19311/stft-why-overlapping-the-window
-->

## Machine learning

Classification is a type of machine learning task where the goal is to
learn a model which can accurately predict which class(es) that data belongs to.
Examples use-cases could be to determine from a image which breed a dog is,
to predict from text whether - or to determine from audio what kind of sound is present.

`TODO: image of a labeled dataset`

In single-label classification, a sample can only belong to a single class. 
In closed-set classification, the possible class is one of N predetermined classes.
Many classification problems are treated as single-label and closed-set.


Models used for classification are often trained using supervised learning. 
Supervised learning uses a dataset where each sample is labeled with the right class.
These labels are normally provided by manual annotation by humans inspecting the data.

```TODO:
describe metrics
```

<!--
SKIP
- class im/balance
-->

![Splitting datasets into train/validation/test sets and cross-validation \label{figure:crossvalidation}](./img/crossvalidation.png)

The dataset is divided into multiple subsets that have different purposes.
The *training* set is data that the training algorithm uses to optimize the model on.
To estimate how well the model generalizes to new unseen data,
predictions are made on the *validation set*. 
The final performance of the trained model is evaluated on a *test set*,
which has not been used in the training process.
To get a better estimate of how the model performs it K-fold cross validation can be used,
where K different training/validation splits are attempted.
K is usually between 5 and 10.
The overall process is illustrated in Figure \ref{figure:crossvalidation}.

One common style of supervised learning processes is to start with an initial model with
some parameters, make predictions using this model, compare these prediction with the labels to compute
an error, and then update the parameters in order to attempt to reduce this error.
This iterative process is illustrated in \ref{figure:training-inference}.

The exact nature of the model parameters and parameter update algorithm
depends on what kind of classifier and training system is used.
For neural networks, see chapter `TODO: ref gradient descent`.

In addition to the parameters learned by the training process,
there are also *hyperparameters* which are parameters that cannot be learned by the training process,
Examples are settings of the training system itself, but also the
architecture and settings of predictive model itself can be seen as hyperparameters.
Hyperparameters can be chosen by selecting different candidates,
training to completion and evaluating performance on the validation set.

![Relationship between training system and the predictive model being trained. \label{figure:training-inference}](./img/training-inference.png)

Once training is completed, the predictive model can be used stand-alone, using the learned parameters.


\newpage
## Audio Classification


### Mel-spectrogram

For machine learning it is desirable to reduce the dimensions of inputs as much as possible.
A STFT spectrogram often has considerable correlation between adjacent frequency bins,
and is often reduced to 30-128 frequency bands using a filter-bank.
Several different filter-bank alternatives have been investigated for audio classification tasks,
such as 1/3 octave bands, the Bark scale, Gammatone and the Mel scale.
All these have filters spacing that increase with frequency, mimicking the human auditory system.
See Figure \ref{figure:filterbanks}

![Comparison of different filter responses. Mel, Gammatone, 1/3-octave \label{figure:filterbanks}](./pyplots/filterbanks.png)


The most commonly used for audio classification is the Mel scale.
The spectrogram that results for applying a Mel-scale filter-bank is often called a Mel-spectrogram.
`TODO: references`

`TODO: image of mel-spectrogram`

Mel-Filter Cepstral Coefficients (MFCC) is a feature representation
computed by performing a Discrete Cosine Transform (DCT) on a mel-spectrogram.
This further reduces dimensionality to just 13-20 bands and reduces correlation between each band.
This has been shown to work well for speech, but can perform worse on general sound classification tasks.
`TODO: reference Stowell birds`

### Normalization

Audio has a very large dynamic range.
The human hearing has a lower threshold of hearing down to $20\mu\text{Pa}$ (0 dB SPL)
and a pain threshold of over 20 Pa (120 dB SPL), a difference of 6 orders of magnitude[@smith1997scientist, ch.22].
A normal conversation may be 60 dB SPL and a pneumatic drill 110 dB SPL, a 4 orders of magnitude difference.
It is common to compress the range of values in spectrograms by applying a log transform.

In order to center the values, the mean (or median) of the spectrogram is often removed.
Scaling the output to a range of 0-1 or -1,1 is also sometimes done.
These changes have the effect of removing amplitude variations,
forcing the model to focus on the patterns of the sound regardless of amplitude.

`TODO: image of normalized mel-spectrogram. Or feature distribution of datasets w/without normalization?`


### Analysis windows

When recording sound, it forms a continuous, never-ending stream of data.
The machine learning classifier however generally needs a fixed-size feature vector. 
Also when playing back a finite-length recording, the file may be much longer than
the sounds that are of interest.
To solve these problems, the audio stream is split up into analysis windows of fixed length,
typically with a length a bit longer than the target sound.
The windows can follow each-other with no overlap,
or move forward by a number less than the window length (overlap).
With overlap a target sound will couple of times, each time shifted.
This can improve classification accuracy. 

![Audio stream split into fixed-length analysis windows](./img/analysis-windows.png)

A short analysis window has the benefit of reducing the feature size of the classifier,
which uses less memory and possibly allows to reduce the model complexity,
which may in turn allow to make better use of a limited dataset. 

When the length of audio clips is not evenly divisible by length of analysis windows,
the last window is zero padded.

### Weak labeling

Sometimes there is a mismatch between the desired length of analysis window,
and the labeled clips available in the training data.
For example a dataset may consist of labeled audio clips with a length of 10 seconds,
and the desired analysis window be 1 second.
When a dataset is labeled only with the presence of a sound at a coarse timescale,
without information about where exactly the relevant sound(s) appears.
it is referred to as *weakly annotated* or *weakly labeled* data.

If one assumes that the sound of interest occur throughout the entire audio clip,
a simple solution is to let each analysis window inherit the label of the audio clip as-is.

If this assumption is problematic, the task can be approached as a Multiple Instance Learning (MIL) problem.
Dedicated MIL techniques for audio classification have been explored in the literature.
`TODO: reference MMM etc`


### Aggregating analysis windows

When evaluating on a test-set where audio clips are 10 seconds,
but the model classifies analysis windows of 1 second
the individual predictions must be aggregate into one prediction for the clip.

A simple technique is *majority voting*,
where the overall prediction is the class that occurs most often across individual predictions.

With *probabilistic voting* (or global mean pooling),
the probabilities of individual predictions are averaged together,
and the output prediction the class with overall highest probability.

`TODO: image showing how voting is performed` 


### Data augmentation

<!--
Motivation
Principle
(General examples)
Commonly used Data Augmentation for audio/ESC
-->

Access to labeled samples is often a limited, because it is expensive to acquire.
This can be a limiting factor for reaching good performance using supervised machine learning.

Data Augmentation is a way to synthetically generate new labeled samples from existing ones,
in order to expand the effective training set.
A simple form of data augmentation can be done by modifying the sample data slightly
in a way such that the class of the sample is still the same.

Common data augmentation techniques for audio include Time-shift, Pitch-shift and Time-stretch.
These are demonstrated in Figure \ref{figure:dataaugmentations}.

![Common data augmentations for audio demonstrated on a dog bark ("woof woof"). Parameters exaggerated to show the effects more clearly. \label{figure:dataaugmentations}](./pyplots/dataaugmentations.png)


Mixup[@Mixup] is another type of data augmentation technique
where two samples from a different classes are mixed together to create a new sample.
A mixup ratio $\lambda$ controls how much the sample data is mixed,
and the labels of the new sample is a mix of labels of the two inputs samples.

$$
\begin{aligned}
\tilde{x} &= \lambda x_i + (1 - \lambda)x_j & \text{  where } x_i, x_j \text{are raw input vectors} \\
\tilde{y} &= \lambda y_i + (1 - \lambda)y_j & \text{  where } y_i, y_j \text{are labels one-hot encoded}
\end{aligned}
$$


The authors argue that this encourages the model to behaving linearly in-between training examples.
It has been shown to increase performance on audio tasks[@ESC-mixup][@AclNet][@Mixup-ASC].

Data augmentation can be applied either to the raw audio waveform,
or to preprocessed spectrograms.

<!--
Other data augmentation:

Frequency response change
Dynamic range compression
Cutout
-->

\newpage
## Convolutional Neural Networks

`TODO: intro, why are they important`

### Neural Networks

`TODO: describe Fully connected layer`

`TODO: describe multi-layer`

`TODO: describe non-linear activation functions`

### Training

`TODO: describe Backpropagation`

`TODO: describe Gradient Decent`

### Convolution

Convolution operation
Functions. Edge detection, median filtering
Depth. Higher-level features. Patterns of patterns

A convolution filter (also called kernel) allows to express many common transformations
on 1D or 2D data, like edge detection (horizontal/vertical) or smoothening filters (median). 
Kernels can be seen as parametric local feature detector can express more complex problem-specific
patterns, like a upward or downward diagonal from a bird chirp when applied to a spectrogram.

Using a set of kernels in combination can detect many pattern variations.

`TODO: image of typical CNN. VGG ?`

<!--
[@BatchNormalization]
-->

### Convolutions in 2D

Moves spatially across the width and height of input.
Convolution is

![Standard 3x3 convolutional block, input/output relationship. Imae: Yusuke Uchida[@ConvolutionsIllustrated]](./img/conv-standard.png)

The computational complexity of such a convolution is $ O_conv = WHNK_wk_hM $,

```
H input height
W input width
N input channels
M output channels
K_w kernel width
K_h kernel height
```

### Pooling

`TODO: describe Max, mean pooling` 

`TODO: image of pooling`

### Strided convolution

Striding can be used to reduce spatial dimensionality, either as an alternative or compliment max/mean-pooling.

? vunerable to aliasing?

`TODO: image of striding`

<!--
"Fully convolutional neural networks". Only Conv operations
Used by ResNet.
-->

### Depthwise Separable convolution

![Depthwise separable convolutions, input/output relationship. Image: Yusuke Uchida[@ConvolutionsIllustrated]](./img/conv-depthwise-separable.png)

While a regular convolution performs a convolution over both channels and the spatial extent,
a Depthwise Separable convolution splits this into two convolutions.
First a Depthwise convolution over the spatial extent,
followed by a a Pointwise convolution over the input channels.
The pointwise convolution is sometimes called a 1x1 convolution,
since it is equivalent to a 2D convolution operation with a 1x1 kernel. 

$$ O_{pw} = HWNM $$
$$ O_{dw} = HWNK_wK_h $$
$$ O_{ds} = O_pw + O_dw = HWN(M + K_wK_h) $$

This factorization requires considerably fewer computations compared to full 2D convolutions.
For example, with $K_w=K_h=3$ and $M=64$, the reduction is approximately $7.5x$.

<!--
Used in [@Xception]
-->

### Spatially Separable convolution

In a spatially separable convolution, a 2D convolution is factorized into two convolutions with 1D kernels.
First a 2D convolution with $1 x K_h$ kernel is performed, followed by a 2D convolution with a $K_w x 1$ kernel. 

$$
O_{ss} = HWNMK_w + HWNMK_h = HWNM(K_w+K_h)
$$

This reduces the number of computations and parameters over regular 2D convolutions
by a ratio $(K_w+K_h)/(K_wK_h)$.
With $K_w=K_h=3$, the reduction is 6/9 and with $K_w=K_h=5$ it is 10/25. 


`TODO: image of spatially separable convolution`


<!---
SKIP
- Residual connections/networks
- Grouped convolutions. HWNK²M/G
? Global Average Pooling
-->

\newpage
## Microcontrollers

`TODO: couple of extra references`

A microcontroller is a tiny computer integrated on a single chip,
containing CPU, RAM, persistent storage (FLASH) as well
as peripherals for communicating with the outside world.

Common forms of peripherals include General Purpose Input Output (GPIO) for digital input/output,
Analog to Digital (ADC) converter for analog inputs,
and high-speed serial communications for digital inter-system communication
using protocols like I2C and SPI.
For digital audio communication, specialized peripherals exists using the I2S or PDM protocols.

Microcontrollers are widely used across all forms of electronics,
from household electronics and mobile devices, telecommunications infrastructure,
cars and industrial systems.
In 2017 over 25 billion microcontrollers were shipped,
and is expected to grow by more than 50% over the next 5 years[@ICInsightsMCUSales].

Examples of application processors from ST Microelectronics
that could be used for audio processing is shown in Table \ref{table:microcontrollers}.
Similar offerings are available from other manufacturers such as
Texas Instruments, Freescale, Atmel, Nordic Semiconductors, NXP.

\begin{table}
\input{pyincludes/microcontrollers.tex}
\caption{Examples of available STM32 microcontrollers and their characteristics. Details from ST Microelectronics website. }
\label{table:microcontrollers}
\end{table}

<!--

ARM expects ML-enabled devices (both using microcontrollers and microprocessors)
to grow 10 times over 300 million units in 2018, and up to 3.2 billion in 2028.
"A backward glance and a forward view" 2018

![Internals of a STM32F103VGT6 microcontroller. Large uniform areas are RAM and FLASH, bottom lower corner has the CPU core. Source: Zeptobars.com[@STM32F103decap]](./img/STM32F103VGT6-LD.jpg)

-->

### Machine learning on microcontrollers

For sensor systems the primary usecase for Machine Learning
is to train a model on a desktop or cloud system ("off-line" learning),
then to deploy the model to the microcontroller to perform inference.
Dedicated tools are available for converting models
to something that can execute on a microcontroller,
usually integrated with established machine learning frameworks.

CMSIS-NN by ARM.
A low-level library for ARM Cortex-M microcontrollers implementing basic neural network building blocks,
such as 2D convolutions, pooling and Gated Recurrent Units.
It uses optimized fixed-point maths and SIMD instructions,
which can be 4x faster and energy efficient than floating point[@CMSISNN].

![Low level functions provided by CMSIS-NN (light gray) for use by higher level code (light blue)[@CMSISNN]](./img/CMSIS-NN-functions.png)

uTensor[@uTensor] by ARM. Allows to run a subset of TensorFlow models on ARM Cortex-M devices,
designed for use with the mbed software platform.

TensorFlow Lite for Microcontrollers, an experimental port of
TensorFlow[@TensorFlow], announced at TensorFlow Developer Summit in March 2019[@LaunchingTensorflowLiteMicrocontrollers].
Its goal is to be compatible with TensorFlow Lite (for mobile devices etc),
and reuse platform-specific libraries such as CMSIS-NN or uTensor in order to be as efficient as possible.

EdgeML by Microsoft Research India[@EdgeMLGithub].
Contains novel algorithms developed especially for microcontrollers,
such as Bonsai[@Bonsai], ProtoNN[@ProtoNN] and FastGRNN[@FastGRNN].

emlearn[@emlearn] by the author.
Supports converting a subset of Scikit-Learn[@scikit-learn] and Keras[@Keras] models
and run them using C code designed for microcontrollers.

X-CUBE-AI[@X-CUBE-AI] by ST Microelectronics provides official support for inference with
Neural Networks for their STM32 microcontrollers. 
It is an add-on to the STM32CubeMX software development kit,
and allows loading trained models from various formats, including:
Keras (Tensorflow), Caffe[@Caffe] and PyTorch.
In X-CUBE-AI 3.4, all computations are done in single-precision float.
Model compression is supported by quantizing model weights by 4x or 8x,
but only for fully-connected layers (not convolutional layers)[@X-CUBE-AI-manual, ch 6.1].
X-CUBE-AI 3.4 does not use CMSIS-NN.


### Hardware accelerators for neural networks

With the increasing interest in deploying neural networks on low-power microcontrollers,
dedicated hardware acceleration units are also being developed.

STMicroelectronics (ST) has stated that neural network accelerators will be available
for their STM32 family of microcontrollers[@ST-DCNN-accelerator], based on their
FD-SOI chip architecture[@ST-FD-SOI].

![Architecture of Project Orlando by ST, system-on-chip with DSP and hardware accelerators for Machine Learning integrated with microcontroller (gray) [@ST-Orlando-MPSoc17]](./img/ST-Orlando-SoC.png){ height=20% }

ARM has announced ARM Helium, an extended instruction set for the Cortex M
family of microcontrollers that can be used to speed up neural networks[@ARMHeliumAnnouncement].

Kendryte K210 is a microcontroller based on the open RISC-V architecture
that includes a convolutional neural network accelerator[@KendryteK210Datasheet]. 

GreenWaves GAP8 is a RISC-V chip with 8 cores designed for parallel-processing.
They claim a 16x improvement in power efficiency over a ARM Cortex M7 chip[@GAP8vsARM].



\newpage
## Environmental Sound Classification

### Datasets
\label{chapter:datasets}

The Urbansound taxonomy[@UrbanSound8k, ch 2] is a proposed taxonomy of sound sources,
developed based on analysis of noise complaints in New York city between 2010 and 2014.
The same authors also compiled the Urbansound dataset[@UrbanSound8k, ch 3],
based on selecting and manually labeling content from the Freesound[@Freesound] repository.
10 different classes from the Urbansound taxonomy were selected and
1302 different recordings were annotated, for a total of 18.5 of labeled audio. 
A curated subset with 8732 audio clips of maximum 4 seconds is known as *Urbansound8k*.

YorNoise[@medhat2017masked] is a collection of vechicle noise.
It has a total of 1527 samples, in two classes: road traffic (cars, trucks, busses) and rail (trains).
The dataset follows the same design as Urbansound8k,
and can be used standalone or as additional classes to Urbansound8k.

ESC-50[@ESC-50] is a small dataset of environmental sounds,
consisting of 2000 samples across 50 classes from 5 major categories.
The dataset was compiled using sounds from Freesound[@Freesound] online repository.
A subset of 10 classes is also proposed, often called ESC-10.
Human accuracy was estimated to be to 81.30% on ESC-50 and
95.7% on ESC-10[@ESC-50, ch 3.1].
The Github repository for ESC-50[@ESC-50-Github] contains a comprehensive summary
of results on the dataset, with over 40 entries.
As of April 2019, the best models achieve 86.50% accuracy,
and all models with over 72% accuracy use some kind of Convolutional Neural Network.

AudioSet [@AudioSet] is a large general purpose ontology of sounds with 632 audio event classes.
The accompanying dataset has over 2 million annotated clips based on sound from Youtube videos.
Each clip is 10 seconds long. 527 classes from the ontology are covered.

In DCASE2019 challenge (in progress, ends July 2019) task 5[@DCASE2019Task5]
audio clips containing common noise categories are to be tagged.
The tagging is formulated as a multi-label classification on 10 second clips.
The dataset[@SONYC-UST] has 23 fine-grained classes across 8 categories
with 2794 samples total.
The data was collected from SONYC noise sensor network in New York city. 

Several earlier DCASE challenge tasks and datasets
have been on related topics to Environmental Sound Classification,
such as Acoustic Scene Detection[@DCASE2017Task1],
general-purpose tagging of sounds[@DCASE2018Task2],
and detection of vehicle related sounds[@DCASE2017Task4].


<!--
TODO: table with dataset statistics

TUT Sound Events 2017[@TUT2017dataset] is a dataset used for the task
"Sound event detection in real life audio" for the
Detection and Classification of Acoustic Scenes and Events (DCASE) challenge in 2017.
-->


### Spectrogram-based models

Many papers have used Convolutional Neural Networks (CNN) for Environmental Sound Classification.
Approaches based on spectrograms and in particular log-scaled melspectrogram being the most common.

PiczakCNN[@PiczakCNN] in 2015 was one of the first applications of CNNs to the Urbansound8k dataset.
It uses 2 channels of log-melspectrograms, both the plain spectrogram values
and the first-order difference (delta spectrogram).
The model uses 2 convolutional layers, first with size 57x6 (frequency x time) and then 1x3,
followed by two fully connected layers with 5000 neurons each.
The paper evaluates short (950ms) versus long (2.3 seconds)
analysis windows, and majority voting versus probability voting.
Performance on Urbansound8k ranged from 69% to 73%.
It was found that probability voting and long windows perform slightly better.
 
![Architecture of Piczak CNN, from the original paper [@PiczakCNN]. \label{figure:piczak-cnn}](./img/piczak-cnn.png)

SB-CNN[@SB-CNN] (2016) is a 3-layer convolutional with uniform 5x5 kernels and 4x2 max pooling.
The paper also analyzes the effects of several types of data augmentation on Urbansound8k.
including Time Shift, Pitch Shift, Dynamic Range Compression and Background Noise.
With all augmentations, performance on their model raised from 72% to 79% classification accuracy.
However time-stretching and pitch-shifting were the only techniques that
consistent gave a performance boost across all classes.


D-CNN[@D-CNN] (2017) uses feature representation and model architecture that largely follows that of PiczakCNN,
however the second layer uses dilated convolutions with a dilation rate of 2. 
With additional data augmentation of time-stretching and noise addition,
this gave a performance of up to 81.9% accuracy on Urbansound8k.
LeakyRelu was found to perform slightly better than ReLu which scored 81.2%.

A recent paper investigated the effects of mixup for data augmentation (2018)[@ESC-mixup].
Their model uses 4 blocks with 2 convolutional layers each followed by max pooling.
The second and third blocks form a spatially separated convolution,
second block with 2 3x1 convolutions, and third block with 2 1x5 convolutions. 
On mel-spectrograms the model scored 74.7% on Urbansound8k without data augmentation,
77.3% with only mixup applied,
and 82.6% when time stretching and pitch shift was combined with mixup.
When using Gammatone spectrogram features instead of mel-spectrogram
performance increased to 83.7%, which seems to be state-of-the-art as of April 2019.


### Audio waveform models 

Recently approaches that use the raw audio waveform as input have also been documented.

![EnvNet[@EnvNet] architecture, using raw audio as input. \label{figure:envnet}](./img/envnet.png)

EnvNet[@EnvNet] (2017) used 1D convolutions in order to learn a 2D spectrogram-like representation
which is then classified using standard 2D convolutional layers.
The architecture is illustrated in Figure \ref{figure:envnet}.
They show that the resulting spectrograms have frequency responses with
a shape similar to mel-spectrograms.
The model manages a 66.3% accuracy score on Urbansound8k[@EnvNet2] with raw audio input.


In [@VeryDeepESC], authors evaluated a number of deep CNNs using only 1D convolutions.
Raw audio with 8kHz sample rate was used as the input.
Their 18 layer model (M18) got a 71% accuracy on Urbansound8k,
and the 11 layer version (M11) got 69%.

EnvNet2[@EnvNet2] (2018) is like EnvNet but with 13 layers total instead of 7,
and using 44.1 kHz input samplerate instead of 16kHz.
Without data augmentation it achieves 69.1% accuracy on Urbansound8k.
When combining data augmentation with a technique similar to mixup called between-class examples,
the model is able to reach 78.3% on Urbansound8k.


## Resource efficient Convolutional Neural Networks

### Environmental Sound Classification

There are also a few works on Environmental Sound Classification (ESC)
that explicitly target making resource efficient models, measured
in number of parameters and compute operations.

WSNet[@WSNet] is a 1D network on raw audio designed for efficiency.
It proposes a weight sampling approach for efficient quantization of weights to
reache an accuracy of 70.5% on UrbandSound8k with a 288K parameters and 100M MAC.

LD-CNN[@LD-CNN] is a more efficient version of D-CNN.
In order to reduce parameters the early layers use spatially separable convolutions,
and the middle layers used dilated convolutions.
As a result the model has 2.05MB of parameters, 50x fewer than D-CNN,
while accuracy only dropped by 2% to 79% on Urbansound8k.

AclNet [@AclNet] is a CNN architecture.
It uses 2 layers of 1D strided convolution as a FIR decimation filterbank
to create a 2D spectrogram-like set of features.
Then a VGG style architecture with Depthwise Separable Convolutions is applied.
A width multiplier ala that of Mobilenet is used to adjust model complexity.
Data augmentation and mixup is applied, and gave up to 5% boost.
Evaluated on ESC-50, the best performing model gets 85.65% accuracy, very close to state-of-the-art.
The smallest model had 7.3M MACs with 15k parameters and got 75% accuracy on ESC-50.

eGRU[@eGRU] demonstrates an Recurrent Neural Network based on a modified Gated Recurrent Unit.
The feature representation used was raw STFT spectrogram from 8Khz audio.
The model was tested using Urbansound8k, however it did not use the pre-existing folds and test-set,
so the results may not be directly comparable to others.
With full-precision floating point the model got 72% accuracy.
When running on device using the proposed quantization technique the accuracy fell to 61%.

As of April 2019, eGRU was the only paper that could be found for the ESC task
and the Urbansound8k dataset on a microcontroller.


### Image classification

The development of more efficient Convolutional Neural Networks for
image classification have received a lot of attention over the last few years.
This is especially motivated by the ability to run models
that give close to state-of-the-art performance on mobile phones and tablets.
Since spectrograms are 2D inputs that are similar to images, it is possible that some of these
techniques can transfer over to Environmental Sound Classification.

SqueezeNet[@SqueezeNet] (2015) focused on reducing the size of model parameters.
It demonstrated AlexNet[@AlexNet]-level accuracy on ImageNet challenge using 50x fewer parameters,
and the parameters can be compressed to under 0.5MB in size compared to 240MB for AlexNet.
It replaced most 3x3 convolutions in a convolution block with 1x1 convolutions,
and reduce the number of channels using "Squeeze" layers consisting only of 1x1 convolutions.
The paper also found that a residual connection between blocks increased model performance
by 2.9% without adding parameters.

Mobilenets[@Mobilenets] (2017) focused on reducing inference computations by
using Depthwise separable convolutions.
A family of models with different complexity was created using two hyperparameters:
a width multiplier $\alpha$ (0.0-1.0) which adjusts the number of filters in each convolutional layer,
and the input image size.
On ImageNet, MobileNet-160 $\alpha=0.5$ with 76M MAC performs better than SqueezeNet with 1700M MAC,
a 22x reduction. The smallest tested model was 0.25 MobileNet-128, with 15M mult-adds and 200k parameters.

![Convolutional blocks of Effnet, ShuffleNet and Mobilenet. Illustration based on Effnet paper[@Effnet]](./img/conv-blocks-imagenets.png)

Shufflenet[@Shufflenet] (2017) uses group convolutions in order to reduce computations.
In order to mix information between different groups of convolutions it introduces
a random channel shuffle.

SqueezeNext[@SqueezeNext] (2018) is based on SqueezeNet but
uses spatially separable convolution (1x3 and 3x1) to improve inference time.
While the MAC count was higher than MobileNet, they claim better inference
time and power consumption on their simulated hardware accelerator.

Effnet[@Effnet] (2018) also uses spatial separable convolutions,
but additionally performs the downsampling in a separable fashion:
first a 1x2 max pooling after the 1x3 kernel,
followed by 2x1 striding in the 3x1 kernel.
Evaluated on CIFAR10 and Street View House Numbers (SVHN) datasets
it scored a bit better than Mobilenets and ShuffleNet. 

### Speech detection

Speech detection is a big application of audio processing and machine learning.
In the Keyword Spotting (KWS) task the goal is to detect a keyword or phrase that
indicates that the user wants to enable speech control.
Example phrases in commercially available products include "Hey Siri" for Apple devices
or "OK Google" for Google devices.
This is used both in smart-home devices such as Amazon Alexa, as well as smartwatches and mobile devices.
For this reason keyword spotting on low-power devices and microcontrollers
is an area of active research.

In [@sainath2015convolutional] (2015) authors evaluated variations of
small-footprints CNNs for keyword spotting. They found that using large strides in time or frequency 
could be used to create models that were significantly more effective.

In the "Hello Edge"[@HelloEdge] paper (2017),
different models were evaluated for keyword spotting on microcontrollers.
Included were most standard deep learning model architectures
such as Deep Neural Networks (DNN), Recurrent Neural Networks and Convolutional Neural Networks.
They found that Depthwise Separable Convolutional Neural Network (DS-CNN) provided the best
accuracy while requiring significantly lower memory and compute resources than other alternatives.
Models were evaluated with three different performance limits.
Their "Small" version with under 80KB, 6M ops/inference achieved 94.5% accuracy on the Google Speech Command dataset.
A DNN version was demonstrated on a high-end microcontroller (ARM Cortex M7 at 216 Mhz) using CMSIS-NN framework,
running keyword spotting at 10 inferences per second while utilizing only 12% CPU (rest sleeping).

FastGRNN[@FastGRNN] (2018) is a Gated Recurrent Neural Network designed
for fast inference on audio tasks on microcontrollers.
It uses a simplified gating architecture with residual connection,
and uses a three-stage training schedule that
forces weights to be quantizated in a sparse and low-rank fashion. 
When evaluated on Google Speech Command Set (12 classes),
their smallest model of 5.5 KB achieved 92% accuracy
and ran in 242 ms on a low-end microcontroller (ARM Cortex M0+ at 48 Mhz).




\newpage
# Materials

## Dataset

The dataset used is Urbansound8K, described in chapter \ref{chapter:datasets}.
The 10 classes in the dataset are listed in Table \ref{table:urbansound8k-classes},
and Figure \ref{figure:urbansound8k-examples} shows example audio spectrograms.
    
\begin{table}
\centering
\input{pyincludes/urbansound8k-classes.tex}
\caption{Classes found in the Urbansound8k dataset}
\label{table:urbansound8k-classes}
\end{table}

![Spectrograms of sound clips from Urbansound8k dataset, selected for each class\label{figure:urbansound8k-examples}](./plots/urbansound8k-examples.png)

The dataset comes pre-arranged into 10 folds.
A single fold may contain multiple clips from the same source file,
but the same source file is not used in multiple folds to prevent data leakage.
Authors recommend always using fold 10 as the test set,
to allow easy comparison of results between experiments.

The target sound is rarely alone in the sound clip, and may be in the background,
partially obscured by sounds outside the available classes.
This makes Urbansound8k a relatively challenging dataset.


## Hardware platform

As the microcontroller we have chosen the STM32L476[@STM32L476] from STMicroelectronics.
This is a mid-range device from ST32L4 series of ultra-low-power microcontroller.
It has a ARM Cortex M4F running at 80 MHz, with hardware floating-point unit (FPU)
and DSP instructions. It has 1024 kB of program memory (Flash), and 128 kB of RAM.

For audio input both analog microphone and and digital microphones (I2S/PDM) are supported.
The microcontroller can also send and receive audio over USB.
This allow to send audio data from a host computer
to test that the audio classification system is working as intended.
An SD card interface can be used to store recorded samples to collect a dataset.

To develop for the STM32L476 microcontroller we selected the
SensorTile development kit STEVAL-STLKT01V1[@STEVAL-STLKT01V1].
The kit consists of a SensorTile module, an expansion board, and a portable docking board (not used).

![SensorTile module with functional blocks indicated. Module size is 13.5x13.5mm\label{sensortile-annotated}](./img/sensortile-annotated.jpg)

The SensorTile module (see figure \ref{sensortile-annotated}) contains in addition to the microcontroller: a microphone,
Bluetooth radio chip, and an Inertial Measurement Unit (accelerometer+gyroscope+compass).
An expansion board allows to connect and power the microcontroller over USB.
The ST-Link V2 from a Nucleo STM32L476 board is used to program and debug the device.
The entire setup can be seen in figure \ref{sensortile-devkit}.

![Development setup of SensorTile kit\label{sensortile-devkit}](./img/sensortile-devkit.jpg)

## Software

The STM32L476 microcontroller is supported by STM32CubeMX development package
and the X-CUBE-AI neural network add-on from ST Microelectronics.
Version 3.4.0 of X-CUBE-AI was used. 

![STM32CubeMX application with X-CUBE-AI addon after loading a Keras model](./img/stm32cubeai.png)

A Python commandline script was created to streamline collecting model statistics using X-CUBE-AI,
without having to manually use the STM32CubeMX user interface. See \ref{appendix:stm32convert}.
This tool provides equired Flash storage (in bytes), RAM usage
and CPU usage (in Multiply-Accumulate operations per second, MAC/s) as JSON,
and writes the generated C code to a specified directory.

The training setup is implemented in Python.
The machine learning models are implemented in Keras using the Tensorflow backend,
and are attached be found in the appendices.
To perform feature extraction during training librosa[@librosa] was used.
numpy and Pandas was used for general numeric computations and data management.

The training setup has automated tests made with pytest,
and uses Travis CI to execute the tests automatically for each change.

All the code used is available at \url{https://github.com/jonnor/ESC-CNN-microcontroller}


## Models

### Model requirements

The candidate models must fit the constraints of our hardware platform,
and leave sufficient resources for other parts of an application to run on the device.
To do so, we allocate a maximum 50% of the CPU, RAM, and FLASH to the model.

ST estimates that an ARM Cortex M4F type device uses approximately 9 cycles/MACC[@X-CUBE-AI-manual].
With 80 MHz CPU frequency this is approximately 9 MACC/second at 100% CPU utilization.


|  Resource    | Maximum (50% utilization)   | Desirable    |
| -------      |:---------:|:------------:|
| RAM usage    |   64 kB   | `< 32 kB`    |
| Flash use    |   512 kB  | `< 256 kB`   |
| CPU usage    |   4.5 M MACC/s   | `< 0.5 M MACC/s`  |

Table: Summary of device constraints for machine learning model


`TODO: link model review section`

Models from the existing literature are shown with respect to
the in model constraints \ref{table:urbansound8k-existing-models-logmel}.
Only SB-CNN and LD-CNN are close.

`MAYBE: move perf table to`

\begin{table}
\input{plots/urbansound8k-existing-models-logmel.tex}
\caption{Existing methods and their results on Urbansound8k}
\label{table:urbansound8k-existing-models-logmel}
\end{table}


![Performance of existing CNN models using log-mel features on Urbansound8k dataset. Green region shows the region which satisfies our model requirements.\label{existing-models-perf}](./plots/urbansound8k-existing-models-logmel.png)
`FIXME: plot is clipping text at bottom and top, need more margins`



### Compared models

SB-CNN and LD-CNN are the two best candidates for a baseline model,
being the only two that are close to the desired performance characteristic.
SB-CNN utilizes a CNN architecture similar to the literature on efficient CNN models,
with small uniformly sized kernels (5x5) followed by max pooling. 
LD-CNN on the other hand uses less conventional full-height layers in the start,
with two heads that take both mel-spectrogram and delta-melspectrogram as inputs.
This requires twice as much RAM as a single input, and the convolutions in the CNN
should be able to learn delta-type features if needed. 
For these reasons SB-CNN was used as the base architecture for experiments.

\ref{existing-models-perf}

The baseline model has a few minor modifications from the original SB-CNN model:
Max pooling is 3x2 instead of 4x2. Without this change the layers become negative sized
due to the reduced input feature size (60 mel filter bands instead of 128).
Batch Normalization was added to each convolutional block.

Would like to evaluate the effects of using more computationally efficient
convolutional blocks, in particular depthwise-separable and spatially-separable convolutions.

`TODO: table of models to test, parameters`

`TODO: images of each compared architecture. Overall / convolutional blocks`

Residual connections are not evaluated, as the networks are relatively shallow.
Grouped convolutions are not evaluated, as they are not supported by our version of Keras and X-CUBE-AI.
They were added to TensorFlow very recently[@TensorFlowGroupConvolutionPR].

To get the RAM utilization within limits, striding is used as the downsampling strategy. 
Since the stride in Keras/Tensorflow must be uniform, 2x2 is used instead of 3x2.

<!--

-->

`TODO: write about RAM optimization in X-CUBE-AI`
In the SB-CNN architecture X-CUBE-AI will fuse the layers Conv2D -> BN -> MaxPooling2D 
into a single operation.
This can drastically reduces RAM usage, from `TODO` to...
Unfortunately this optimization is not implemented for all cases. (`FIXME: WHICH`)
`TODO: images of RAM usage per layer`

<!--
Some other models were also attempted.

DenseNet. X-CUBE-AI conversion fails. `INTERNAL ERROR: 'refcount'`
MobileNet. Had to replace Relu6() with ReLu.
EffNet. Had to replace LeakyReLU with ReLu.

ST FP-SENSING1 function pack[@FP-AI-SENSING1]
-->


\newpage
# Methods

![Overview of the full model. The classifier runs on individual analysis windows, and predictions for the whole audio clip done using voting. \label{classification-pipeline}](./img/classification-pipeline.png)

## Preprocessing

Mel-spectrograms is used as the input feature.
The most compact and most computationally efficient featureset in use by existing methods was by LD-CNN,
which used windows of 31 frames @ 22050 Hz (720 ms) with 60 mels bands.
This has achieved results near the state-of-art, so we opted to use the same.


\begin{table}
\centering
\input{pyincludes/experiment-settings.tex}
\caption{Summary of preprocessing and training settings}
\label{table:experiment-settings}
\end{table}


During preprocessing we also perform Data Augmentation.
Time-stretching and Pitch-shifting following [@SB-CNN], for a total of 12 variations per sample.
The preprocessed mel-spectrograms are stored on disk as Numpy arrays for use during training.

During training time each window of mel-spectrogram frames is normalized by subtracting
the mean of the window and dividing by the standard deviation.

## Training

<!--
`?! Include Hyperparameter search ?`
-->

The pre-assigned folds of the Urbansound8k dataset was used,
with 9-fold cross-validation during training and fold 10 as the held-out test set.

Training are done on individual windows,
with each window inheriting the label of the audio clip it belongs to.

In each minibatch, audio clips from training set are selected randomly.
And for each sample, a time window is selected from a random position[@SB-CNN].
This effectively implements time-shifting data augmentation.

In order to evaluate the model on the entire audio clip, an additional
pass over the validation set is done which combines predictions from multiple time-windows
as shown in Figure \ref{classification-pipeline}.

As the optimizer, Stocastic Gradient Decent (SGD) with Nesterov momentum set to 0.9 is used.
Learning rate was set to 0.005 for all models. Each model is trained for up to 50 epochs.
A complete summary of experiment settings can be seen in Table \ref{table:experiment-settings}.

Training was performed on a NVidia GTX2060 GPU with 6GB of RAM to reduce experiment time,
however the models can be trained on any device supported by TensorFlow and a minimum of 1GB RAM.

## Evaluation

Once training is completed, the model epoch with best performance on the validation set is selected
for each of the cross-validation folds.
The selected models are then evaluated on the test set.

In addition to the original Urbansound8k test set,
we also evaluate the models performance on two simplified variations:

- Only clips where target sound is in the foreground
- Grouping into 5 more coarse classes 

`TODO: table of group membership`

The SystemPerformance application skeleton from X-CUBE-AI is used to record the
average inference time per sample on the STM32L476 microcontroller.
This accounts for potential variations in number of MACC/second for different models,
which would be ignored if only relying on the theoretical MACC number. 


\newpage
# Results


![Test accuracy of the different models](./img/models_accuracy.png){ height=30% }

\begin{table}
\input{pyincludes/results.tex}
\caption{Results for the compared models}
\label{table:results}
\end{table}


`TODO: add results with Baseline with Depthwise Separable`

`TODO: add results of different amounts of conv kernels for DS-5x5`

`TODO: add results with Effnet (spatially separable)`

![Accuracy versus compute of different models](./img/models_efficiency.png){ height=30% }

![Confusion matrix on Urbansound8k](./img/confusion_test.png){ height=30% }

![Confusion matrix in reduced groups with only foreground sounds](./img/grouped_confusion_test_foreground.png){ height=30% }

`TODO: add error analysis. Are misclassifications marked as low-confidence?`

`TODO: plot training curves over epochs`

\newpage
# Discussion

<!--
Ref Problem
> Can we classify environmental sounds directly on a wireless and battery-operated noise sensor?
-->

`TODO: make into coherent flow`

The Baseline model uses more CPU than our requirements, as expected.
Also the base Strided model is outside the desirable range.

Depthwise Separable combined with striding (Strided-DS-5x5, Strided-DS-3x3)
able to match the baseline performance, at a much lower computational cost.

Best result 73%. Far from the state-of-the-art when not considering performance constraints
Probably below human-level accuracy. Ref ESC-50

<!--
Almost reaching level of PiczakCNN[@SB-CNN] with data augmentation,
and better than without data augmentation[@PiczakCNN].
With estimated 88M MAC/s, a factor 200x more.
Indicator of huge differences in efficiency between different CNN architectures
-->

When considering only foreground sounds, accuracy increases significantly.

When considering the reduced 5-group classification.
Some misclassifications are within a group of classes, and this increases accuracy.
Example...
However still have significant confusion for some groups...

`TODO: update to reflect latest results`

<!--
SKIP
Possible to use slightly bigger microcontroller.
Able to double Flash. Up to 1024kB RAM, 8x. Approx 8x CPU.

What is the approx cost of system. BOM
What is the battery lifetime. BOM
-->

# Conclusions

Able to demonstrate Environmental Sound Classification
running on a low-power microcontroller suitable for use in a sensor node.

The best model achieves a `` accuracy when evaluated on the Urbansound8k dataset,
using `XX %` of the CPU capacity.

`TODO: evaluate`
??? is the perf high enough to be useful in practice?
??? When considering foreground/grouped, and class of errors

## Further work

CNN quantizations for efficient integer inference. 
[@IncrementalNetworkQuantization]

Use fixed-point / SIMD optimimized CNN implementation.
4-5x speedup possible. Ref [@CMSIS-NN]

It is possible that the performance level of `<75%`
can be matched with a lower sample-rate and fewer mel-filter bands.

Hybrid methods.
Perform classification on-device, but also
allow to send a feature representation.
Maybe for a subset of classification candidates

In the coming years, hardware accelerators for
Convolutional Neural Networks are expected to become available.
Significantly better power efficiency and compute power.


<!---
APPENDIX
TODO: clean up the scripts, make fit on one/two page
MAYBE: table with software versions? From requirements.txt
-->

\begin{appendices}

% introduces custom pythoncode command
% ref https://tex.stackexchange.com/questions/103141/set-global-options-for-inputminted
\newmintedfile[pythoncode]{python}{
fontsize=\footnotesize
}

`TODO: move appendix after references`

\section{Keras model for SB-CNN (Baseline)}
\pythoncode{../microesc/models/sbcnn.py}
\label{listing:sbcnn}

\newpage
\section{Keras model for Strided}
\pythoncode{../microesc/models/strided.py}
\label{listing:ldcnn}

\newpage
\section{Script for converting models using X-CUBE-AI}
\label{appendix:stm32convert}
\pythoncode{../microesc/stm32convert.py}

\end{appendices}

\newpage
# References

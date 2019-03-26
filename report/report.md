
\newpage
# Introduction

Noise
Environmental noise
Health problems
Reduced value
Regulations
Noise assesment
Challenges. Local problem, multiple sources, time-dependent
Increasing noise problem. Urbanization
Positive sound qualities. Recreational. Perceptual, subjective.
Track noise level, plan/take corrective action. Visibility
Identifying source. Environmental Sound Classification
Smart-city concept, data-driven
Wireless Sensor Networks
Existing projects
Privacy, GDPR
=> this thesis



Sound is everywhere around us. It is a communication tool, speech
side-effect of many human activities
construction
machinery


workplace, occupational noise


## Environmental noise


Noise is unwanted sound.


Environmental noise is the summary of noise pollution from outside,
caused by transport, industrial and recreational activities.
Road traffic is the most widespread source of environmental noise in urban environments.



## Regulations
Environmental noise is regulated in the EU by the Environmental Noise Directive (2002/49/EC)[@EuNoiseDirective].
The purpose of the directive is to:

* determine peoples exposure to environmental noise
* ensuring that information on environmental noise and its effects is available to the public
* preventing and reducing environmental noise where necessary
* preserving environmental noise quality where it is good

The Directive requires Member States to prepare and publish noise maps and noise management action plans every 5 years for:

* agglomerations with more than 100,000 inhabitants
* major roads (more than 3 million vehicles a year)
* major railways (more than 30.000 trains a year)
* major airports (more than 50.000 movements a year, including small aircrafts and helicopters)

The Directive does not set limit or target values, nor does it prescribe the measures to be included in the action plans.
This is up to authorities of each individual Member State.

However, Environmental Noise Directive defines *indicators* for noise pollution:

$L_{den}$: Designed to assess overall annoyance.
It refers to an annual average day, evening and night period of exposure.
Evening are weighted 5 dB(A) and a night weighting of 10 dB(A).
Indicator level: 55dB(A).

$L_{night}$: Designed to assess sleep disturbance.
It refers to an annual average night period of exposure.
Indicator level: 50dB(A).

In Norway, the coverning legislation for noise pollution is Forurensningsloven[@Forurensningsloven],
which implements the EU directive.

## Health impact

According to European Commission introduction on the Health effects of Noise [@EuNoiseHealthEffects],
Noise pollution the second environmental cause of health problems in Europe, after air pollution.

Sleepers that are exposed to night noise levels above 40dB on average throughout
the year can suffer health effects like sleep disturbance and awakenings.
Above 55dB long-term average exposure, noise can trigger elevated blood pressure and lead to ischaemic heart disease.
The WHO has set a Night Noise Guideline level for Europe at 40 dB $L_{night}$.

According to a report done on behalf of the European Commision[@RVIMTransportationNoise]

"The exposure to transportation noise in Europe led in 2011 to about
900 thousand cases of hypertension and 40 thousand hospital admissions due to cardiovascular disease and stroke"
and "the number of cases of premature mortality due to these diseases as a result of noise exposure is about 10 thousand per year.",
and "An estimated 8 million people experience sleep disturbance due to transportation noise and about 4 million perceive this as severe".

## Urban noise

    TODO: reference existing research / groups / project / locations


## Noise measurements

Periodic noise measurements can be done with hand-held Sound Level meters.
Their specifications are standardized in IEC 61672-1 Sound Level Meters[@IECSoundLevelMeters].
Use of a handheld device requires an operator to be present, which limits how often
and at how many locations measurements are made.

![Norsonic Nor140 handheld acoustic measurement unit](./images/nor140.jpg)

With a continous noise monitoring station, measurement are be done automatically,
giving very good coverage over time.
Many such stations can be deployed to also give good spatial coverage,
operating together in a Wireless Sensor Network.

![CESVA TA120 noise monitoring station](./images/cesva-ta120.png)

## Existing projects


In addition to commercial products, a number of research projects have deployed sensor networks for acoustic noise.
This includes SONYC[@Sonyc] in New York City, and the Sentilo[@Sentilo] project in Barcelona.

The noise profile data is based on the 1/3 octave band, following the standard IEC 61260-1:2014[@IECOctaveBands].
This can be used by a machine learning system to distinguish different noise sources[@AudioCodingSensorGrid].
The paper also demonstrates that when frequency spectrum samples are performed 10 times per second or more seldom,
it is not possible to understand conversations. This preserves the privacy requirement.



## Noise monitoring with Wireless Sensor Networks

    TODO: write

Continious monitoring.
Low cost. Enables high density of sensor nodes

Wireless connectivity
Energy harvesting / battery operation

On-edge processing
Save energy, bandwidth
Respect privacy

This motivates our research question:

> Can we classify environmental sounds directly on the noise sensor,
without requiring to transmit audio or features to a central system?




# Background

## Sound level
Sound level is measured in decibel (dB).
0dB is the threshold of hearing, at $20 µPascal$ relative sound pressure. 
The level is normally A-weighted, which simulates the frequency response of human hearing.

![Descibel scale with common noise sources](./images/decibel-scale.jpg)

## Equivalent Continious Sound Level
The sound level is constantly changing.
To get a single number representation, the sound level is averaged over a time period **T**.

![Equivalent continious sound level](./images/equivalent-continious-level.jpg)


## Machine learning

Supervised learning
Training set
Validation set
Test set
Cross-fold validation


## Data augmentation

Data augmentation
Pitchshift,timestretch
Mixup,between-class


## Classification

In classification the goal is to determine the which acoustic event appears in an audio sample.
Samples can be short relative to the acoustic event length, or long, possibly having many instances of an event.
The number of events and their time is not returned in classification.

![Classification task with one class label per audio sample. Source: [@ComputationalAnalysisSound, fig 6.1]](./images/classification.png)

The classification can be binary (is this birdsong?), or multi-class (which species is this?).
Classification is often limited to a single label.

When the classification is done on long samples consisting of many different events
it is called Acoustic Scene Classification. Examples of a 'scene' in urban environments
could be 'restaurant', 'park', 'office', each having a different
composition of typical acoustic events.

## Event Detection

In event detection (also known as onset detection) the goal is to find the time spans where a given acoustic event occurs.
If the acoustic event is "frog croaking", then for each instance of a frog croak
the start time and end time of this event should be marked.

![Event detection, with labels at precise start and end locations in time. Source: [@ComputationalAnalysisSound, fig 6.3] ](./images/eventdetection.png)

In monophonic event detection, only the most prominent event is returned.
A classifier ran on short samples relative to the length of the acoustic event
can be used a detector.

In polyphonic event detection, multiple events are allowed at the same time.
This can be approached using separate classifiers per event type,
or using a multi-label classifier as a joint model.

## Weak labeling

Many acoustic events are short in duration, for instance a door slamming.
Other acoustic events only happen intermittently, like the the frog vocalizations in previous example.

Under supervised learning, ideally each and every of the acoustic events instances in the training data
would be labeled with their start and end time. This is called 'strong labels'.
However aquiring strong labels requires careful attention to detail by the annotator and is costly.

Therefore, often per-event time-based labels are not available.
Instead fixed-length audio clips are only marked whether an event occurs at least once, or not at all.
This is called a 'weak label' or 'weakly annotated' data.

When performing event detection, with small time windows (compared to annotated audio clips) as input,
the missing time information means there is not a direct label for this input.
This is known as a Multiple Instance Learning (MIL) problem.
Under MIL input instances are grouped into a 'bag', and the label exists on the bag
instead of the individual instances.
MIL formulations exist for many common machine learning algorithms.

## Digital sound

Physically, sound is a variation in pressure over time.
For machine learning, it must be exist in a digital representation.
The acoustic data is converted to analog electric signals by a microphone and
then digitized using an Analog-to-Digital-Converter (ADC).

    TODO: IMAGE, replace with own work

![From acoustical sound to digital and back. Source: [@ProcessingTutorial]](./images/digital-sound-processingorg.png)

In the digitization process, the signal is quantized in time at a certain sampling frequency,
and the amplitude quantized at a certain bitdepth.
A typical sampling frequency is 44100 Hz, and bitdepth 16 bit. With these parameters,
the acoustic sound can be reconstructed without perceivable differences by humans.

Digital sound can be stored uncompressed (PCM .wav), using lossless compression (.flac)
or using lossy compression (.mp3). Lossy compression removes information and may add artifacts,
and is best avoided for machine learning tasks.
Recordings can be multi-channel but for acoustic events
single-channel (mono) data is still the most common.

## Frames

To process and analyze the audio data it is often represented as *frames*, small groups of samples across time.
Frames can be produced in real-time, by collecting N samples at a time,
or by splitting up audio files after they have been recorded.

In a frame based approach, a frame is the smallest unit of time processed by the machine learning algorithm.
Therefore the frame length must be set long enough to contain enough relevant information,
but not so long that temporal variations disappear. For speech, a typical choice of frame length is 25ms.
Similar frame lengths are often adopted for acoustic events, unless there are specific concerns.

    TODO: IMAGE, replace with own work

![Computing frames from an audio signal, using windows functions. Based on image by [@AudioFraming]](./images/frames.png)

Frames often use overlapping samples at the start and end.
Without overlap, acoustic event that happens partially in one frame results in different signals than when appearing in the middle.
The overlap can be specified as percentage of the frame length (overlap percentage).
or as a number of samples (hop length). Overlap can for instance be 50%.
A window function is applied to ensure that the signal level stays constant also in overlapping sections.

## Short Time Fourier Transform

## Spectrogram filterbanks

A raw Short Time Fourier Transform can contain 1024 or more bins, often with strong correlation across multiple bins.
To reduce dimensionality, the STFT spectrogram is often processed with a filter-bank of 40-128 frequency bands.

Some filter-bank alternatives are 1/3 octave bands, the Bark scale, Gammatone and the Mel scale.

All these have filters spacing that increase with frequency, mimicking the human auditory system.

## Mel-spectrogram

A spectrogram processed with triangular filters evenly spaced on the Mel scale is called a Mel-spectrogram.


![Comparison of different filter responses. Mel, Gammatone, 1/3-octave](./pyplots/filterbanks.png)


    TODO: image of normalized mel-spectrogram

    TODO: image of spectrogram and mel-spectrogram of a sound sample


## Convolutional Neural Network

Convolution operation
Functions. Edge detection, median filtering
Depth. Higher-level features. Patterns of patterns

Pooling



    TODO: reference CNNs as state-of-the-art in ESC

A convolution filter (also called kernel) allows to express many common transformations
on 1d or 2d data, like edge detection (horizontal/vertical) or smoothening filters (median). 
But kernels kernel can be seen as parametric local feature detector can express more complex problem-specific
patterns, like a upward or downward diagonal from a bird chirp when applied to a spectrogram.
Using a set of kernels in combination can detect many pattern variations.

    TODO: IMAGE, replace with own work

![Convolution kernel as edge detector, applied to image. Source: [@UnderstandingConvolution]](./images/convolution.png)

## Efficient CNNs

Depthwise separable convolutions. Depth multiplier "1x1(xD) convolution", pointwise
Spatially separable convolutions.
Dilated convolutions.

Dilated Residual Networks

MobileNets
EffNet



## Windowed voting

Mean
Majority
Overlap


## Microcontrollers

    TODO: write 
    What are microcontrollers
    Where are they used
    Special considerations. Autonomous operation. Low power. Low cost.
    Number of shipments anually


\begin{table}
\input{pyincludes/microcontrollers.tex}
\caption{Examples of available ARM microcontrollers and their characteristics}
\label{table:microcontrollers}
\end{table}

Recommended prices from ST Microelectronics website for 1-10k unit orders.

Similar offerings are available from other manufacturers such as
Texas Instruments, Freescale, Atmel, Nordic Semiconductors, NXP.


### Machine learning on microcontrollers

Inference, not training
General purpose vs
Available tools.
ARM CMSIS-NN
tflite
uTensor
emlearn[@emlearn]
ELL

### Hardware accelerators


Kendryte K120. RISC-V
GreenWaves GAP8

    TODO: get some numbers for TOPS/second

STMicroelectronics has stated that neural network accelerators will be available
for their STM32 family of microcontrollers in 2019`TODO: ref`, based on their
FD-SOI chip architecture[@ST-FD-SOI].

ARM has announced ARM Helium, a set of for the Cortex




\newpage
# Materials

## Hardware platform

As the microcontroller we have chosen the STM32L476[@STM32L476] from STMicroelectronics.
This is a mid-range device from ST32L4 series of ultra-low-power microcontroller.
It has a ARM Cortex M4F running at 80MHz, with hardware floating-point unit (FPU)
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

STM32CubeMx
STM32AI
FP-AI-SENSING1
Keras. Tensorflow backend
librosa
Pandas,numpy
? Docker image
? training on Kubernetes


Function pack


All computations are done in single-precision float.

Supports model compression by quantizing model weights.
8x or 4x.
Tool can perform basic validation of the compressed model. 

Does not support multi-input models.

    TODO: include screenshot of STM32AI




The machine learning models are trained in Python using Keras with Tensorflow backend,
To perform feature extraction 

librosa[@librosa]

    TODO: picture of training + deployment pipelines




TODO: reference CMSIS-NN, ARM Keyword spotting 4x faster using fixed-point/SIMD.

## Model requirements

Our machine learning algorithm must fit on the target device.
By benchmarking 

    TODO: move to methods section?

For RAM and . The remaining 


|  Resource    | Maximum   | Desirable    |
| -------      |:---------:|:------------:|
| RAM usage    |   64 kB   | `< 32 kB`    |
| Flash use    |   512 kB  | `< 256 kB`   |
| CPU usage (MACCs/s)    |   4 M    | `< 0.5 M`    |

Table: Summary of device contraints for machine learning model

    TODO: describe benchmark used to reach MACC/s number


\newpage
## Urbansound8K dataset

The Urbansound8K dataset[@UrbanSound8k] was collected in 2014
based on selecting and manually labeling content from the Freesound[@Freesound] repository.
The dataset contains 8732 labeled sound clips with a total duration of 8.75 hours.
Most clips are 4 seconds long, but shorter clips also exist.
10 different classes are present, as shown in table \ref{table:urbansound8k-classes}.
The classes are a subset of those found in the Urbansound taxonomy,
which was developed based on analysis of noise complaints in New York city between 2010 and 2014.
    
\begin{table}
\centering
\input{pyincludes/urbansound8k-classes.tex}
\caption{Classes found in the Urbansound8k dataset}
\label{table:urbansound8k-classes}
\end{table}

![Spectrograms of sound clips from Urbansound8k dataset, selected for each class\label{urbansound8k-examples}](./plots/urbansound8k-examples.png)

The target sound is rarely alone in the sound clip, and may be in the background, partially obscured by sounds outside the available classes.
This makes Urbansound8k a relatively challenging dataset.
For figure \ref{urbansound8k-examples} sounds with clear occurences of the target sound were chosen.

The dataset comes pre-arranged into 10 folds. A single fold may contain multiple clips from the same source file,
but the same source file is not used in multiple folds to prevent data leakage.
Authors recommend always using fold 10 as the test set, to allow easy comparison of results between experiments.


\newpage
# Methods


## Existing models

    TODO: put both of these into one figure environment?

\begin{table}
\input{plots/urbansound8k-existing-models-logmel.tex}
\caption{Existing methods and their results on Urbansound8k}
\label{table:urbansound8k-existing-models-logmel}
\end{table}

    FIXME: plot is clipping text at bottom and top, need more margins

![Performance of existing CNN models using log-mel features on Urbansound8k dataset. Green region shows the region which satisfies our model requirements.](./plots/urbansound8k-existing-models-logmel.png)


    TODO: refer


SB-CNN
LD-CNN

## Experimental Setup

1. Determine model requirements. Run neural networks, compare runtime vs MACC
Choose most feasible base model. Feature representation


Evaluate effects of:
- Changing model architecture
- Different voting overlap



Preprocessing.
Data augmentation

Hyperparameter search

Cross-validation

Approach towards reaching goals


Model variations

Optimizer
Hyperparameters

Feature extraction settings


\newpage
# Results

    TODO: boxplots of accuracy


    TODO: accuracy versus MACCs


\newpage
# Discussion

would this be good enough to be useful for classifying noise assessment?

might not be neccesary to go as fine-grained as 10 classes
Road noise, people/social noise, construction noise.
could this be done as post-processing on these 10 classes?

could do only the `foreground` classes. Since the predominant sound


class accuracies
confusion 
top3 performance

Further work

Use fixed-point / SIMD optimimized CNN implementation
Using slightly bigger microcontroller.
Able to double Flash. Up to 1024kB RAM, 8x. Approx 8x CPU.


\newpage
# References


<!---

Contributions

- Demonstrate an ESC system running on a 5 USD microcontroller.
Resulting in XX% accuracy on Urbansound8k dataset
- Set of tools for STM32 CubeMX AI to overcome current limitations
in the platform
-->


\newpage
# Introduction

<!---
Sound
    Importance. 

Noise
Sources of noise.
    People talking
    Dogs barking
    Construction
    Air Conditioner, Refridgerator
    
Health problems. Sleep disturbance, 
Economic impact. Reduced value of 
Types of noise. Occuptional. Environmental noise
Regulations


Noise assesment
Challenges.
    Local problem, multiple sources, time-dependent
    perceptual/subjective evaluation
    Positive sound qualities. Recreational
Increasing noise problem. Urbanization

Track noise level, plan/take corrective action. Visibility
Identifying source. Environmental Sound Classification
Smart-city concept, data-driven
Wireless Sensor Networks
Existing projects
Privacy, GDPR
=> this thesis
-->


Sound is everywhere around us, and a rich source of information about our surroundings.

Like other animals, humans communicate 

We use sound explicitly to communicate when we talk, offering observations, facts


## Importance

When we talk, sound is a critical part of our communication, conveying not just statements
but also information about.
As we walk around 

Sound can convey information also when 
before we can see them
alerted about dangerous situation

Sound is used to communicate, be it by speech, audible gestures or computerized signals (

car speeding up or slowing down
fire alarm

We communicate with eachother using sound when we talk, 

communicating with intent

side-effect of many human activities


source of information
understanding of the environment around us


Since the industrial revolution, the human soundscape has become increasingly busy.

mechanical and electromechanical devices
Urbanization. Many more people in the same area

transportation. Cars, railroads, aeroplanes
construction. Drilling, cutting
machinery

The sum of all the noise is referred to as Environmental noise.

Environmental noise is the summary of noise pollution from outside,
caused by transport, industrial and recreational activities.


## Regulation

Environmental noise is a type of noise pollution, and is regulated.

In the EU, Environmental noise is regulated Environmental Noise Directive (2002/49/EC)[@EuNoiseDirective].

The purpose of the directive is to:

* determine peoples exposure to environmental noise
* ensuring that information on environmental noise and its effects is available to the public
* preventing and reducing environmental noise where necessary
* preserving environmental noise quality where it is good

The Directive requires Member States to prepare and publish noise maps and noise management action plans every 5 years for
urban areas and major road, railways and airports.

The Directive does not set limit or target values, nor does it prescribe the measures to be included in the action plans.
This is up to authorities of each individual Member State.


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


## Noise monitoring with Wireless Sensor Networks

Several cities have started to deploy networks of sound sensors in order to understand and reduce noise issues.
These consist of many sensor nodes positioned in the area of interest,
transmitting the data to a central system for storage and reporting.

The SONYC[@SONYC] project in New York City had 56 sound sensors as of 2018.[@SONYC2019]
The Barcelona Noise Monitoring System[@BarcelonaNoiseMonitoring] had 86 sound sensors as of 2018.[@BarcelonaNoiseMonitoring2018].
CENSE[@CENSE] project plans to install around 150 sensors in Lorient, France[@CENSESensor].

To keep costs low and support a dense coverage, the sensor nodes are aften designed to operate wirelessly.
Communication is done using wireless radio technologies such as WiFi, GSM, NB-IoT or 6LoWPAN.
Energy to power the sensor is harvested, either using solar power or from streetlight powered at night.
A battery backup allows the sensor to continue operating also when energy is momentarily unavailable.

These sensor networks enable continious logging of the sound level (Leq dB).
Typical measurement resolution are per-minute, per second or per 125ms.
Sound level sensors in Europe are designed to specifications of IEC 61672-1 Sound Level Meters[@IECSoundLevelMeters],
with an accuracy of either Class 2 or Class 1.
The equivalent standard for North America is ANSI S1.4[@ANSISoundLevelMeters], and Type 1/2 accuracy.

Most sensors also aim to provide information that can be used to characterize the noise.
This requires much more data than sound level measurements,
making it challenging to transmit within the given bandwidth and energy budget of a sensor.
Recording and storing detailed audio data may also capture sensitive information and violate privacy requirements.

To address these concerns

Sending the raw audio waveform 

A simple method is to transmit and store the raw audio, however this is problematic
fro

The noise profile data is based on the 1/3 octave band, following the standard IEC 61260-1:2014[@IECOctaveBands].
This can be used by a machine learning system to distinguish different noise sources[@AudioCodingSensorGrid].
The paper also demonstrates that when frequency spectrum samples are performed 10 times per second or more seldom,
it is not possible to understand conversations. This preserves the privacy requirement.



 and get data needed for efficiently managing and reducing noise. 

On-edge processing

Save energy, bandwidth
Respect privacy


## Problem statement

The need for low-cost and widely deployed noise monitoring solutions
that can inform not only about the sound level but also give information about the noise source,
motivates our research question:

> Can we classify environmental sounds directly on a wireless and battery-operated noise sensor?


\newpage
# Background

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

Identifying noise source

    Problem formulations. Classification

        Out-of-scope
            - Sound directivity.
            - Multi-channel audio. Microphone arrays
            - Multi-sensor fusion

        Approach
            **Classification**. Closed-set. Single label.
            Not open-set classification / clustering
            Not event detection. Not fine identification in time (onset)

    Machine Learning.
        Supervised
        Data representations

    Environmental Sound Classification (ESC).
    Definition
    Datasets
        Urbansound8k
        ESC-50
        DCASE...
    Existing work
    Related.
        Acoustic Scene Classification. Context-dependent compute
        Domestic Sound Classification. 
    DCASE conference and challenge

    Convolutional Neural Networks


-->

## Measuring noise

## Sound level
The amplitude of a soundwave is specified by the variation in pressure, measured in pascal (Pa).
Because of the wide range of possible values, Sound Pressure Level (SPL) is normally specified
using the logarithmic decibel (dB) scale. A reference level of $20 µPascal$ is often used
as the 0dB point. This is an estimate of the threshold of human hearing with air as the medium.

    TODO: equation for converting 
    TODO: explain reason for A-weighting.
    TODO: plot of A weighting frequency response
    TODO: reference standard defining A-weigthing 

The level is normally A-weighted, which simulates the frequency response of human hearing.

    TODO: own picture of noise sources on dB scale

![Descibel scale with common noise sources](./images/decibel-scale.jpg)

## Equivalent Continious Sound Level
The sound pressure level is constantly changing.
To get a single number representation, the sound level is averaged over a time period **T**.

    TODO: mention other measurements, like L10/L90, Lpeak, Lmax
    TODO: put in some equations

![Equivalent continious sound level](./images/equivalent-continious-level.jpg)

## Sound Level Meters

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

## Windowed voting

Mean
Majority
Overlap


## Efficient CNNs

Depthwise separable convolutions. Depth multiplier "1x1(xD) convolution", pointwise
Spatially separable convolutions.
Dilated convolutions.

Dilated Residual Networks

MobileNets
EffNet





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

Due to the constraints of microcontroller hardware,
most of the traditional machine learning frameworks cannot be used directly. 
Instead dedicated tools are available for this niche, usually integrating with established frameworks.

CMSIS-NN by ARM.
A low-level library for ARM Cortex-M microcontrollers implementing basic neural network building blocks,
such as 2D convolutions, pooling and Gated Recurrent Units.
It uses optimized fixed-point maths and SIMD instructions,
which can be 4x faster and energy efficient than floating point[@CMSISNN].

uTensor[@uTensor] by ARM. Allows to run a subset of TensorFlow models on ARM Cortex-M devices,
designed for use with the mbed software platform.

TensorFlow Lite for Microcontrollers, an experimental port of
TensorFlow was announced at TensorFlow Developer Summit in March 2019[@LaunchingTensorflowLiteMicrocontrollers].
Its goal is to be compatible with TensorFlow Lite (for mobile devices etc),
and reuse platform-specific libraries such as CMSIS-NN or uTensor in order to be as efficient as possible.

EdgeML by Microsoft Research India[@EdgeMLGithub].
Contains novel algorithms developed especially for microcontrollers,
such as Bonsai[Bonsai], ProtoNN[@ProtoNN] and FastGRNN[@FastGRNN].

emlearn[@emlearn] by the author.
Supports converting a subset of Scikit-Learn[@scikit-learn] and Keras[@Keras] models
and run them using C code designed for microcontrollers.

### Hardware accelerators


Kendryte K120. RISC-V
GreenWaves GAP8

    TODO: get some numbers for TOPS/second

STMicroelectronics has stated that neural network accelerators will be available
for their STM32 family of microcontrollers in 2019`TODO: ref`, based on their
FD-SOI chip architecture[@ST-FD-SOI].

ARM has announced ARM Helium, an extension for the Cortex M
family of microcontrollers with instructions that can be used to speed up neural networks. `TODO:ref `



\newpage
# Materials

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

The STM32L476 microcontroller is supported by STM32CubeMX`TODO: ref` development package from ST Microelectronics.
ST also provides the X-CUBE-AI`TODO: ref` addon for STM32CubeMX, which provides integrated support for Neural Networks.
In this work, X-CUBE-AI version 3.3.0 was used. 


The addon allows loading trained models from various formats, including:
Keras (Tensorflow), Caffe and PyTorch.

![STM32CubeMX application with X-CUBE-AI addon after loading a Keras model](./img/stm32cubeai.png)

X-CUBE-AI supports model compression by quantizing model weights. Available settings for compression are 4x or 8x.
However as of version 3.4.0, the compression is applied only to fully-connected layers (not to convolutional layers)[@X-CUBE-AI-manual, ch 6.1].
The tool can perform basic validation of the compressed model. 
As of version 3.4.0, all computations are done in single-precision float.

Example code from the FP-AI-SENSING1 Function pack`TODO: ref` from ST was used as a skeleton for the device code.
This implements mel-spectrogram feature pre-processing in C, and uses the model code output by X-CUBE-AI to perform inference.

`MAYBE: table with software versions? From requirements.txt`

The training setup is implemented in Python.
The machine learning models are implemented in Keras using the Tensorflow backend.
To perform feature extraction during training librosa[@librosa] was used.
numpy and Pandas is used for

The training software has automated tests made with pytest,
and uses Travis CI to execute the tests automatically for each change.

All the code used is available at https://github.com/jonnor/ESC-CNN-microcontroller.

`TODO: picture of training + deployment pipelines`



## Model pipeline

`TODO: image of pipeline`



\newpage
# Methods


## Determining model requirements 

The models to be evaluated must fit the hardware constraints of the target device.
From the [Microcontrollers](#microcontrollers) section, these constraints are primarily:
CPU execution time, RAM memory and Flash memory. 

\begin{table}
\input{plots/urbansound8k-existing-models-logmel.tex}
\caption{Existing methods and their results on Urbansound8k}
\label{table:urbansound8k-existing-models-logmel}
\end{table}

`TODO: ref attachments for Keras models`

To evaluate the existing models reviewed in `TODO: link section` several were implemented in Keras,
and ran them though STM32 Cube AI tool to get the required Flash storage (in bytes), RAM usage
and CPU usage (in Multiply-Accumulate operations per second, MACC/s).

A Python commandline script was developed to streamline collecting these statistics and outputting the C code for the model,
without having to manually use the CubeMX user interface. See \ref{appendix:stm32convert}.

The results can be seen in \ref{table:urbansound8k-existing-models-logmel}.

To determine the approximate number of MACC/s that our target hardware is able to sustain,
a few model variations with different complexity (MACC/s) were created.
The SB-CNN model was used as a base, with 30 mels bands (the default in ST FP-SENSING1 function pack).

Then the models were ran on on device (with random inputs), and their execution time recorded.
The results can be seen in. From this we estimate.

`TODO: table of MACC/s and execution time`
`TODO: compute estimated CPU capacity`

\ref{existing-models-perf}


    FIXME: plot is clipping text at bottom and top, need more margins

![Performance of existing CNN models using log-mel features on Urbansound8k dataset. Green region shows the region which satisfies our model requirements.\label{existing-models-perf}](./plots/urbansound8k-existing-models-logmel.png)



|  Resource    | Maximum   | Desirable    |
| -------      |:---------:|:------------:|
| RAM usage    |   64 kB   | `< 32 kB`    |
| Flash use    |   512 kB  | `< 256 kB`   |
| CPU usage    |   4 M MACC/s   | `< 0.5 M MACC/s`  |

Table: Summary of device contraints for machine learning model

## Models to evaluate

Model families:

SB-CNN
LD-CNN
MobileNet
EffNet


## Preprocessing

Mel-spectrograms is used as the input feature.
The most compact and most computationally efficient featureset in use by existing methods was by LD-CNN,
which used 31 frames @ 22050 Hz with 60 mels bands.
This achieved results near the state-of-art, so we opted to use the same.

`TODO: table with preprocessing settings`

During preprocessing we also perform Data Augmentation.
Time-stretching and Pitch-shifting following `TODO: ref SB-CNN`

The preprocessed mel-spectrograms are stored as compressed Numpy arrays.

Each window of mel-spectrogram frames is normalized by subtracting
the mean of the window and dividing by the standard deviation.

## Training

`?! Include Hyperparameter search ?`

10-fold cross-validation using the pre-assigned folds of the Urbansound8k dataset and fold 10 as the held-out test set.

Training is done with minibatches of size of `TODO`.
In each batch, audio clips from training set are selected randomly.
And for each sample, a time window is selected from a random position.
`TODO: ref SB?`

As the optimizer, Stocastic Gradient Decent (SGD) with Nesterov momentum set to 0.9 is used.
Learning rate of `TODO`.
Each model is trained for 50 epochs.

Training was performed on a NVidia GTX2060 GPU with 6GB of RAM to reduce experiment time,
however the models can be trained on a CPU supported by TensorFlow and a minimum of 1GB RAM.


## Evaluation

Once training is completed, the model with best perfomance on the validation set is selected.
The selected models are then evaluated on the test set.

In addition to the unmodified Urbansound8k test set,
we also evaluate the models on two simplified variations:

- Only clips where target sound is in the foreground
- Reduction into 3 classes: 

The testset is also execution on device, recording the average inference time per sample. 

<!---
TODO
9. Measure current draw
(10. Try different voting overlaps)
-->


\newpage
# Results

    TODO: boxplots of accuracy


    TODO: accuracy versus MACCs


\newpage
# Discussion

Problem
> Can we classify environmental sounds directly on a wireless and battery-operated noise sensor?

<!--
What is the approx cost of system. BOM
What is the battery lifetime. BOM
-->

would this be good enough to be useful for classifying noise assessment?

might not be neccesary to go as fine-grained as 10 classes
Road noise, people/social noise, construction noise.
could this be done as post-processing on these 10 classes?

could do only the `foreground` classes. Since the predominant sound


class accuracies
confusion 
top3 performance



Further work

Reduce number of mels or number of frames.
Use fixed-point / SIMD optimimized CNN implementation
Using slightly bigger microcontroller.
Able to double Flash. Up to 1024kB RAM, 8x. Approx 8x CPU.


<!---
APPENDIX
TODO: make be after references
TODO: clean up the scripts, make fit on one/two page
-->

\begin{appendices}

% introduces custom pythoncode command
% ref https://tex.stackexchange.com/questions/103141/set-global-options-for-inputminted
\newmintedfile[pythoncode]{python}{
fontsize=\footnotesize
}


\section{SB-CNN Keras model}
\pythoncode{../microesc/models/sbcnn.py}
\label{listing:sbcnn}

\newpage
\section{LD-CNN Keras model}
\pythoncode{../microesc/models/ldcnn.py}
\label{listing:ldcnn}

\newpage
\section{MobileNet Keras model}
\label{appendix:mobilenet}
\pythoncode{../microesc/models/mobilenet.py}

\newpage
\section{Script for converting models using X-CUBE-AI}
\label{appendix:stm32convert}
\pythoncode{../microesc/stm32convert.py}

\end{appendices}

\newpage
# References

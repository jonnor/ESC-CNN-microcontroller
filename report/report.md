
\newpage
# Introduction

Privacy, GDPR
Wireless Sensor Networks

Supervised machine learning
Neural Networks
Convolutional Nets

## Environmental noise
Noise is unwanted sound. Environmental noise is the summary of noise pollution from outside,
caused by transport, industrial and recreational activities.
Road traffic is the most widespread source of environmental noise in urban environments.

## Sound level
Sound level is measured in decibel (dB).
0dB is the threshold of hearing, at $20 µPascal$ relative sound pressure. 
The level is normally A-weighted, which simulates the frequency response of human hearing.

![Descibel scale with common noise sources](./images/decibel-scale.jpg)

## Equivalent Continious Sound Level
The sound level is constantly changing.
To get a single number representation, the sound level is averaged over a time period **T**.

![Equivalent continious sound level](./images/equivalent-continious-level.jpg)


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

Specifically for workplace monitoring, and evaluating risk of hearing impairment
noise dosimeters are also used. They are standardized in IEC 61252 Personal Sound Exposure Meters[@IECPersonalSoundExposureMeters].
These are not used for evaluating environmental noise.

![Cirrus Research DoseBadge5 noise dosimeter](./images/dosebadge5.jpg)

With a continous noise monitoring station, measurement are be done automatically,
giving very good coverage over time.
Many such stations can be deployed to also give good spatial coverage,
operating together in a Wireless Sensor Network.

![CESVA TA120 noise monitoring station](./images/cesva-ta120.png)


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

## Spectrograms

A raw Short Time Fourier Transform can contain 1024 or more bins, often with strong correlation across multiple bins.
To reduce dimensionality, the STFT spectrogram is often processed with a filter-bank of 40-128 frequency bands.
Some filter-bank alternatives are 1/3 octave bands, the Bark scale, Constant-Q transform and the Mel scale.
All these have filters spacing that increase with frequency, mimicking the human auditory system.

## Mel-spectrogram

A spectrogram processed with triangular filters evenly spaced on the Mel scale is called a Mel-spectrogram.

    TODO: IMAGE, replace with own work

![Mel-spaced filterbank. Filters are set to to be unity-height. Mel-filters using unit-area filters also exist. Source: [@SpeechProcessingTutorial]](./images/mel-filterbanks-20.png)

![Mel-spectrogram of birdsong. The birdsong is clearly visible as up and down chirps at 3kHz and higher](./images/bird_clear_melspec.png)

A mel-spectrogram can still have significant correlation between bands.

## Convolution

A convolution filter (also called kernel) allows to express many common transformations
on 1d or 2d data, like edge detection (horizontal/vertical) or smoothening filters (median). 
But kernels kernel can be seen as parametric local feature detector can express more complex problem-specific
patterns, like a upward or downward diagonal from a bird chirp when applied to a spectrogram.
Using a set of kernels in combination can detect many pattern variations.

    TODO: IMAGE, replace with own work

![Convolution kernel as edge detector, applied to image. Source: [@UnderstandingConvolution]](./images/convolution.png)

## Convolutional Neural Network


    TODO: reference CNNs as state-of-the-art in


## Wireless Sensor Networks


## Microcontrollers

    TODO: write 

STM32 AI


\newpage
# Materials

## Hardware platform

    Key specifications.
    ARM Cortex M4F
    80 Mhz

    STM32AI
    SensorTile


## Datasets

Urbansound taxonomy and Urbansound8K dataset[@UrbanSound8k].

The taxonomy was based on analysis of noise complaints in New York city between 2010 and 2014.

    TODO: describe dataset collection. Where/how??


The dataset consists of  of 10 different classes.

\input{report/pyincludes/urbansound8k-classes.tex}

    TODO: IMAGE with representative spectrogram for each of the classes

The dataset comes pre-arranged into 10 folds.
As recommended by the authors we fold 10 as the test set.
This allows comparison with existing results in literature that do the same.



    TODO: Use and describe ESC-50 dataset


\newpage
# Methods

SB-CNN

\newpage
# Results

\newpage
# Discussion

\newpage
# References

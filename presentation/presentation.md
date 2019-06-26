
---
author: Jon Nordby <jononor@gmail.com>
date: June 26, 2019
margin: 0
css: style.css
pagetitle: 'ss'
---

# Intro { data-state="intro" data-background="img/cover.png" }

<!--
### Environmental Sound Classification on Microcontrollers using Convolutional Neural Networks
-->

# Introduction

## Me

Internet of Things specialist

- B.Eng in **Electronics**
- 9 years as **Software** developer. **Embedded** + **Web**
- M. Sc in **Data** Science

## Problem statement

> Can we classify environmental sounds directly
> on a wireless and battery-operated noise sensor?

## Noise Pollution

Reduces health due to stress and loss of sleep

In Norway

* 1.9 million affected by road noise (2014, SSB)
* 10'000 healty years lost per year (Folkehelseinstituttet)

In Europe

* 13 million suffering from sleep disturbance (EEA)
* 900'000 DALY lost (WHO)


::: notes

1.9 million
https://www.ssb.no/natur-og-miljo/artikler-og-publikasjoner/flere-nordmenn-utsatt-for-stoy

1999: 1.2 million 

10 245 tapte friske leveår i Norge hvert år
https://www.miljostatus.no/tema/stoy/stoy-og-helse/


https://www.eea.europa.eu/themes/human/noise/noise-2

Burden of Disease WHO
http://www.euro.who.int/__data/assets/pdf_file/0008/136466/e94888.pdf

:::


## Noise Mapping

Simulation only, no direct measurements

![](img/stoykart.png)

::: notes

- Known sources
- Yearly average value
- Updated every 5 years
- Low data quality. Ex: communal roads

Image: https://www.regjeringen.no/no/tema/plan-bygg-og-eiendom/plan--og-bygningsloven/plan/kunnskapsgrunnlaget-i-planlegging/statistikk-i-plan/id2396747/

:::

## Noise Monitoring

![](img/noise-monitoring.jpg)

Measures the noise level continiously

::: notes

* No network cabling, no power cabling
* No site infrastructure needed
* Less invasive
* Fewer approvals needed
* Temporary installs feasible
* Mobile sensors possible

Electrician is 750 NOK/hour

Image: https://www.nti-audio.com/en/applications/noise-measurement/unattended-monitoring
:::

## Wireless Sensor Networks

- Wide and dense coverage wanted
- Sensors need to be low-cost
- **Opportunity**: Wireless reduces costs
- **Challenge**: Power consumption


## Environmental Sound Classification

> Given an audio signal of environmental sounds,
> determine which class it belongs to

* Widely researched. 1000 hits on Google Scholar
* Datasets. Urbansound8k (10 classes), ESC-50, AudioSet (632 classes)
* 2017: Human-level performance on ESC-50

::: notes

https://github.com/karoldvl/ESC-50

:::

## Architecture

![](../report/img/sensornetworks.png){width=100%}

## Microcontroller 

![](../report/img/sensortile-annotated.jpg){width=100%}

::: notes

STM32L476

ARM Cortex M4F
Hardware floating-point unit (FPU)
DSP SIMD instructions
80 MHz CPU clock 
1024 kB of program memory (Flash)
128 kB of RAM.

:::

## Model requirements

With 50% of STM32L476 capacity:

* 4.5 M MACC/second
* 64 kB RAM
* 512 kB FLASH memory

::: notes

* RAM: 1000x 64 MB
* PROGMEM: 1000x 512 MB
* CPU: 1000x 5 GFLOPS
* GPU: 1000'000X 5 TFLOPS

:::


# Existing work

- Convolutional Neural Networks dominate
- Mel-spectrogram input standard
- End2end models: getting close in accuracy
- Techniques come from image classification
- "Edge ML" focused on mobile-phone class HW
- "Tiny ML" (sensors) just starting

::: notes

* Efficient Keyword-Spotting
* Efficient (image) CNNs
* Efficient ESC-CNN

ESC-CNN

* 23 papers reviewed in detail
* 10 referenced in thesis
* Only 4 consider computational efficiency

:::


## Urbansound8k methods

![](../report/plots/urbansound8k-existing-models-logmel.png){width=100%}

eGRU: running on ARM Cortex-M0 microcontroller, accuracy 61% with **non-standard** evaluation

::: notes

Assuming no overlap. Most models use very high overlap, 100X higher compute

:::

## Depthwise-separable

MobileNet, "Hello Edge", AclNet

![](../report/img/depthwise-separable-convolution.png){width=100%}
 
## Spatially-separable

EffNet, LD-CNN

![](../report/img/spatially-separable-convolution.png){width=100%}



# Materials

- Dataset: Urbansound8k
- Hardware: STM32L476 SensorTile
- Inference framework: X-CUBE-AI
- Models: CNN, 11 variations
- Training: Keras+Tensorflow

## Urbansound8k 

![](../report/plots/urbansound8k-examples.png){width=100%}


## X-CUBE-AI

![](img/xcubeai.png)

::: notes

* First public version came out in December 2018
* Version 3.4.0
* Not reported in use yet on Google Scholar
* Ahead of TensorFlow Lite for Microcontrollers
* Means did not need to implement myself in `emlearn`
* Decisive for choosing the STM32 hardware platform 

:::

## Models

<!--
Based on SB-CNN (Salamon+Bello, 2016)
-->

![](../report/img/models.svg){height=100%}


## Convolution options

`TODO: image of different convolutions blocks`


## All models

![](img/models-list.png)

::: notes

* Baseline is outside requirements
* Rest fits the theoretical constraints
* Sometimes had to reduce number of base filters to 22 to fit in RAM

:::


# Methods

Standard procedure for Urbansound8k

- Classification problem
- 4 second sound clips
- 10 classes
- 10-fold cross-validation, predefined
- Metric: Accuracy

## Pipeline

![](../report/img/classification-pipeline.png){max-height=100%}

## Preprocessing

- Convert to log-Mel-spectrogram
- 22 kHz, 60 mels, 1024 bin FFT, 31 frame window (720 ms)
- Time-stretching and Pitch-shifting data augmentation
- 12 variations per sample

<!--
![](../report/results/training-settings.png)
-->

## Training

- NVidia RTX2060 GPU 6 GB
- 10 models x 10 folds = 100 training jobs
- 100 epochs
- 3 jobs in parallel
- 36 hours total

::: notes 

- ! GPU utilization only 15%
- CPU utilization was near 100%
- Larger models to utilize GPU better?
- Parallel processing limited by RAM of biggest models
- GPU-based augmentation might be faster

:::

## Evaluation

For each fold of each model

1. Select best model based on validation accuracy
2. Calculate accuracy on test set

For each model

- Measure CPU time on device

# Results & Discussion

## Model comparison

![](../report/results/models_accuracy.png){width=100%}

::: notes

- Baseline relative to SB-CNN and LD-CNN is down from 79% to 73%
Expected because poorer input representation.
Much lower overlap 

:::

## List of results

![](img/results.png){width=100%}

## Performance vs compute

![](../report/results/models_efficiency.png){width=100%}

:::

- Performance of Strided-DS-24 similar to Baseline despite 12x the CPU use
- Suprising? Stride alone worse than Strided-DS-24
- Bottleneck and EffNet performed poorly
- Practical speedup not linear with MACC

:::

## Spectrogram processing

Mel-spectrogram preprocessing<br/>(30 mels, 1024 FFT, 16 kHz)<br/>**60** milliseconds


Stride-DS-24 model<br/>(60 mels, 1024 FFT, 22 kHz)<br/>**81** milliseconds


::: notes

* Bottleneck for reducing CPU time / power consumption
* Opportunity for end2end models.
* Does not seem to be there yet? Not explicitly considered in literature
* Especially interesting with CNN co-processors

:::

## Confusion

![](../report/results/confusion_test.png){height=100%}

## Conclusions

- Best architecture: Depthwise-Separable convolutions with striding
- Best performance: 70.9% mean accuracy, under 20% CPU
- Highest reported Urbansound8k on microcontroller (over eGRU 62%)
- Spectrogram preprocessing becoming a bottleneck
- Feasible to perform ESC on microcontroller


# Demo

## Demo video

`TODO: add inline or link`



# Next steps

## Improving performance

Model quantization

- *CMSIS-NN* 8bit SIMD -> 4x speedup

Stronger training process 

- Data Augmentation. *Mixup*, *SpecAugment*
- Transfer Learning on more data. *AudioSet*

Evaluate 16 kHz, 30 mels

:::
EnvNet-v2 got 78.3% on Urbansound8k with 16 kHz
:::

## Practical challenges

- Real-life performance evaluations. Out-of-domain samples
- Reducing power consumption. Adaptive sampling
- Efficient training data collection in WSN. Active Learning? 


## Soundsensing

![](img/soundsensing-logo.png)

Sensor Systems for Noise Monitoring

- Supported by Norwegian Research Council
- Pilot project with Oslo Kommune
- Accepted to incubator at StartupLab

# Summary

## Summary

- Noise pollution is a growing problem
- Wireless Sensor Networks can used to quantify
- On-sensor classification desirable for power/cost and privacy
- Thorough literature review on efficient CNN and ESC
- Methods. Follows established practices for ESC
- Results. Best reported for Urbansound8k on a microcontroller
- Demonstrated working in a practical test
- Basis for a new company from NMBU Data Science

# Questions? { data-state="questions" data-background="img/cover.png" }


# BONUS



## Grouped classification

![](../report/results/grouped_confusion_test_foreground.png){}

Foreground-only

## Adding Unknown class

![](img/unknown-class.png){}

::: notes

Idea: If confidence of model is low, consider it as "unknown"

* Left: Histogram of correct/incorrect predictions
* Right: Precision/recall curves
* Precision improves at expense of recall
* 90%+ precision possible at 40% recall

Usefulness:

* Avoids making decisions on poor grounds
* "Unknown" samples good candidates for labeling->dataset. Active Learning 
* Low recall not a problem? Data is abundant, 15 samples a 4 seconds per minute per sensor

:::

## CPU efficiency

![](img/cpu-efficiency.png){}

::: notes


:::

## What could be done better

- Hyperparameter tuning per model
- Run evaluation also on unmodified SB-CNN,
verify training process
- Fix random seeds,
ensure exact reproducability
- Request review feedback sooner and more often,
catch mistakes earlier

## What worked well

- Automated pipeline for result processing.
Easy updating of new results into report
- Using official X-CUBE-AI instead of own implementation,
or using less-ready inference library
- Automated checking of device constraints,
eased finding suitable model parameters

<!--
## Publishing

MAYBE: add quote or teaser

- Planning a paper submission to journal/conference
- Have contacted Salamon, Bello
-->


# MISC

## Mel-spectrogram

![](../report/img/spectrograms.svg)

## 2D-convolution

![](../report/img/convolution-2d.png){width=100%}

## Downsampling using max-pooling

![](../report/img/maxpooling.png){width=100%}

## Downsampling using strided convolution

![](../report/img/strided-convolution.png){width=100%}

## Bug: Integer truncation

![](img/fail-truncation.png)

## Dropout location

![](img/fail-dropout.png)




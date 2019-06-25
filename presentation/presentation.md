
---
title: Environmental Sound Classification on Microcontrollers using Convolutional Neural Networks
author: Jon Nordby <jononor@gmail.com>
date: June 26, 2019
margin: 0
css: style.css
---

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

`TODO: describe impacts`

## Environmental Sound Classification

::: notes

Widely researched. 1000 hits on Google Scholar

:::

## Wireless Sensor Networks

- Wide and dense coverage needed -> Sensors need to be low-cost
- Opportunity: Wireless simplifies installation, reduces costs
- Challenge: Power consumption

::: notes
No network cabling, no power cabling
No site infrastructure needed
Less invasive
Fewer approvals needed
Temporary installs feasible
Mobile sensors possible

Electrician is 750 NOK/hour
:::

## Data Transmission

![](../report/img/sensornetworks.png){width=100%}

## Microcontroller 

STM32L476

![](../report/img/sensortile-annotated.jpg){width=100%}

::: notes

ARM Cortex M4F
Hardware floating-point unit (FPU)
DSP SIMD instructions
80 MHz CPU clock 
1024 kB of program memory (Flash)
128 kB of RAM.

:::

## Model constraints

`TODO: list constraints`

::: notes

- Training and inference are independent implementations. Have to be in sync

:::

# Existing work


## Audio Classification state-of-the-art

- Convolutional Neural Networks dominate
- Mel-spectrogram input standard
- End2end models (raw audio input) research very active


## Mel-spectrogram

![](../report/img/spectrograms.svg){width=100%}

## Existing Urbansound methods

![](../report/plots/urbansound8k-existing-models-logmel.png){width=100%}

::: notes

Assuming no overlap. Most models use very high overlap, 100X higher compute

:::

## Depthwise-separable convolution

MobileNet, "Hello Edge"

![](../report/img/depthwise-separable-convolution.png){width=100%}
 
## Spatially-separable convolution

EffNet, LD-CNN

![](../report/img/spatially-separable-convolution.png){width=100%}



# Materials

## Overview

- Dataset: Urbansound8k
- Hardware: STM32L476 SensorTile
- Inference framework: X-CUBE-AI
- Models: CNN, 11 variations
- Training: Keras+Tensorflow

## Urbansound8k 

![](../report/plots/urbansound8k-examples.png){width=100%}


## X-CUBE-AI

`TODO: add screenshot`

::: notes

First public version came out in December 2019
Version 3.4.0 used
Not reported in use yet on Google Scholar

:::

## Models

<!--
Based on SB-CNN (Salamon+Bello, 2016)
-->

![](../report/img/models.svg){height=100%}


## Convolution options

`TODO: image of different convolutions`



# Methods

- Classification
- Couple of

## Preprocessing

## Training

- 10 models x 10 folds = 100 training jobs
- 3 jobs in parallel, 36 hours total

## Evaluation

- Measuring CPU time on device

`TODO: screenshot`

# Results & Discussion

## Model comparison

![](../report/results/models_accuracy.png){width=100%}

## Performance vs compute

![](../report/results/models_efficiency.png){width=100%}

## Spectrogram processing time

`TODO: add results`

## Confusion matrix

![](../report/results/confusion_test.png){width=100%}

## Conclusions

`TODO: list conclusions`

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

## Practical challenges

- Real-life performance evaluations. Out-of-domain samples
- Reducing power consumption. Adaptive sampling
- Efficient training data collection in WSN. Active Learning? 


## Soundsensing

`TODO: add company logo`

- Supported by Norwegian Research Council
- Pilot project with Oslo Kommune
- Accepted by StartupLab incubator

# Summary

## Summary

- Noise pollution is a growing problem
- Wireless Sensor Networks used to quantify
- On-sensor classification desirable for power/cost and privacy
- Thorough literature review on efficient CNN and ESC
- Methods. Follows established practices for ESC
- Results. Best reported for Urbansound8k on a microcontroller
- Demonstrated working in a practical test
- Basis for a new company from NMBU Data Science

# Questions?

`TODO: add project image`

# BONUS

## Unknown class

Improves precision at expense of recall

`TODO: add picture`

## Grouped classification

![](../report/results/grouped_confusion_test_foreground.png){}

Foreground-only

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

- Planning a paper submission to journal/conference
- Have contacted Salamon, Bello
-->

## Working method

- Independent study Audio Classification. DAT390
- DCASE2018 conference
- Literature review. Summarized in report
- Experiments phase. Tried large amount of different classifiers

# MISC

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

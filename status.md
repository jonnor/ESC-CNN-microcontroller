---
title: Audio Classification using general-purpose microcontrollers
author: Jon Nordby <jonnord@nmbu.no>
date: February 07, 2019
---

## Overall

**GANTT chart**
https://jonnor.github.io/thesis-audio-classification-microcontrollers/plan.html

Summary

- On schedule
- Experimentation going a bit slow


## Tasks for Feburary 07

Done

- Identify and reproduce some baseline models on datasets
- Define & Explore some hypotheses for better models
- Run a custom model on STM32 microcontrolle

## Run custom model on microcontroller

**YouTube video**

Hickups

- STM32 NN implementation is propriatary
- Pre-processing code hardcoded to 30 mels/ 1024 FFTs
- Pre-processing only non-overlapping analysis windows
- Example Python script *WRONG...*

Findings

- **Inference time spent on pre-processing is significant**

# Baseline models

## Speech Commands

Google Speech Commands dataset

- Tensorflow CNN models tested & reproduced.
- Got accuracy and training time improvement by changing to log-mel
- ARM. DS-GNN
- Microsoft Research. FastGRNN

## Environmental 

ESC-50

* Best models achieving 86.50% accuracy.
* Human accuracy estimated 81.30%.
* Baseline CNN at 64.50%. 
* Baseline MFCC-RF, 44.30%.

No models on microcontrollers found!


## Hypotheses

**On github**

https://github.com/jonnor/thesis-audio-classification-microcontrollers/blob/master/braindump.md#hypotheses


# Next

## Tasks for February 21

- Explore/narrow down hypotheses more
- Ideally choose promising models
- Write methods
- Write background

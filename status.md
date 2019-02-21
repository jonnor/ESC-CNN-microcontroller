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

- Explore hypotheses
- Choose promising models
- Write background. **6 pages**
- Write methods

# Model exporation


## Urbansound8k

- SB-CNN 
- DenseNet
- Dilated Convolutions
- FastGRNN.

## Experimental setup

* Samplerate 16kHz
* 32 band log-mel
* Checking 1 of 9-fold CV
* 1 second windows
* Oversampling dataset, 7 pieces per 4 seconds (50% overlap)
* Mean/majority voting over

## SB-CNN

Baseline:

- 44.1kHz, 3 second window
- Voting over 1-frame overlap
- Picking best epoch using voted validation
- 72% average on 9 fold CV

Mine

- Best model: **1 fold**.
- 73% test. 65% val??
- 9-fold CV result pending

## Dilated

! Missing info in paper, number of kernels per layer. Email sent

Best model: 65% single frame

## DenseNet

Missing info in paper

- No details about DenseNet comparison model
- Email sent

!!! DenseNet models faile to validate in STM32CubeMXAI.

Best model: 67% single frame 

## Findings

- Small dataset. Sensitive to overfitting. Extra work needed to compensate. Oversampling dataset
- All methods seem to top out at roughly same. Bottleneck on training process?
- Good: A variation of SB-CNN looks feasible from compute perspective

## FastGRNN

- Ran example code
- No Keras/STM32 support
- TF micro is experimental
- Looks hard to run micro. Downprioritize?

Best model: N/A


# Next

## Tasks for March 7

- Write methods
- Finish writing background
- Fix/add data augmentation
- Setup validation pipeline

# Misc

## ESC-50

* Best models achieving 86.50% accuracy.
* Human accuracy estimated 81.30%.
* Baseline CNN at 64.50%. 
* Baseline MFCC-RF, 44.30%.

## Hypotheses

**On github**

https://github.com/jonnor/thesis-audio-classification-microcontrollers/blob/master/braindump.md#hypotheses


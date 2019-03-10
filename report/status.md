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


## Tasks for March 7

- Write methods
- Finish writing introduction
- Fix/add data augmentation
- Setup validation pipeline

# Writing

## Introduction

- Outline clarified
- 25% done?

## Background

- Outline. 20%
- Some non-mathy stuff there

## Materials

In pretty good shape.

- Dataset DONE
- Hardware platform DONE
- Software platform 50%
- Existing methods 50%


# Experiments 

## Unhardcoded mel-spec processing

* Tools for generating Hann window + Mel filters lookup-tables
* Not yet ran on device

## Validation pipeline

- OK
- Can run easily on Google Cloud Engine. But still slow!! CPU utilization low
- Showing voted accuracy included during training, easier to evaluate
- SB-CNN up to 72% accuracy

Learned

- Sensitive to hyperparameters!

## LD-CNN

- Good: Was able to get it to run, with single input.

LD-CNN. Best model: 79% voted, 90% overlap
Using augmentation.

SB-CNN is 78% with data augmentation


# Next

## Tasks for March 21

- Write methods
- Finish writing introduction
- Write in existing results
- Do Hyperparameter search
- Do Model search


## Hypotheses

**On github**

https://github.com/jonnor/thesis-audio-classification-microcontrollers/blob/master/braindump.md#hypotheses


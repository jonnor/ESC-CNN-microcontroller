
## Title

Efficient audio classification using general-purpose microcontrollers

## Problem statement

How to make the most (cost,power) efficient audio classification
for real-time use on microcontroller?

Scope:

Focused on Convolutional Neural Networks as a base/framework

## Methodology

- Select a general purpose microcontroller. Having a specific amount of storage,RAM and CPU.

STM32 family spans a large range.
Mid-range. STM32L4 low-power (ARM Cortex M4F+).
High perf. STM32F4, STM32F7.
Low-perf. STM32F1/STM32L1
Support for standard microphone,LogMel,CNNs out-of-the-box 
Note: most usecases require (wireless) connectivity.
Can use separate chip for demo?
SensorTile devkit has Bluetooth chip included.

- Select open datasets for audio classification
Acoustic event detection,
Acoustic noise source classification,
Acoustic Scene classification,
Keyword spotting
Speech command
Should have at least one binary classification, and one multi-label. Maybe detection/segmentation
Should have one with short events, and one with longer effects "scene" etc

- Select a couple of baseline methods
With reference results available on 
Ideally with open code for easily reproducability.

- Create and tests hypothesis for approaches to take for making more efficient models


Power efficiency. Measurement: Current consumption. Proxy: CPU inference time 
Cost efficient. Measurement: Microcontroller cost. Proxy: RAM,CPU requirements

## Plans

- Problem statement reviewed


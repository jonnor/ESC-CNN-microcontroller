
# Environmental Sound Classification on Microcontrollers using Convolutional Neural Networks

## Thesis report

<a href="https://github.com/jonnor/ESC-CNN-microcontroller/releases/download/print1/report-print1.pdf"><img src="https://github.com/jonnor/ESC-CNN-microcontroller/raw/master/report/img/frontpage.png" height="200" alt="Download thesis report (PDF)"></a>

<a href="https://github.com/jonnor/ESC-CNN-microcontroller/releases/download/print1/report-print1.pdf">Download thesis report (PDF)</a>

### Errata

In `print1` version of report

- Fig 5.1. DS-Strided-24 result is missing
- Fig 5.1. No-information-rate should be 11.5% instead of 10%.
Did not take class-imbalance into account
- Fig 2.10. Labels EffNet and ShuffleNet swapped
- Fig 5.3. Missing description of model used. Uses Stride-DS-24
- Table 4.1. Nesterov momentum shows NaN. Should be 0.9

## Citing

You can use the following BibTeX entry

```bibtex
@mastersthesis{esc_micro_cnn_nordby2019,
    title={Environmental Sound Classification on Microcontrollers using Convolutional Neural Networks},
    author={Jon Nordby},
    year=2019,
    month=5,
    school={Norwegian University of Life Sciences},
    url={http://hdl.handle.net/11250/2611624}
}
```

## Keywords

    Wireless Sensor Networks, Embedded Systems
    Edge Computing, Edge Machine Learning
    Noise classification, Environmental Sound Classification (ESC), Urbansounds
    Tensorflow, Keras, librosa

## Abstract

Noise is a growing problem in urban areas,
and according to the WHO is the second environmental cause of health problems in Europe.
Noise monitoring using Wireless Sensor Networks are
being applied in order to understand and help mitigate these noise problems.
It is desirable that these sensor systems, in addition to logging the sound level,
can indicate what the likely sound source is.
However, transmitting audio to a cloud system for classification is
energy-intensive and may cause privacy issues.
It is also critical for widespread adoption and dense sensor coverage that
individual sensor nodes are low-cost.
Therefore we propose to perform the noise classification on the sensor node,
using a low-cost microcontroller.

Several Convolutional Neural Networks were designed for the
STM32L476 low-power microcontroller using the Keras deep-learning framework,
and deployed using the vendor-provided X-CUBE-AI inference engine.
The resource budget for the model was set at maximum 50% utilization of CPU, RAM, and FLASH.
10 model variations were evaluated on the Environmental Sound Classification task
using the standard Urbansound8k dataset.

The best models used Depthwise-Separable convolutions with striding for downsampling,
and were able to reach 70.9\% mean 10-fold accuracy while consuming only 20% CPU.
To our knowledge, this is the highest reported performance on Urbansound8k using a microcontroller.
One of the models was also tested on a microcontroller development device,
demonstrating the classification of environmental sounds in real-time.

These results indicate that it is computationally feasible to classify environmental sound
on low-power microcontrollers.
Further development should make it possible to create wireless sensor-networks
for noise monitoring with on-edge noise source classification.


## Run experiments

Install dependencies

    pip install -r requirements.txt

Preprocess audio files into features

    python3 preprocess.py

Train the models

    python3 jobs.py

Evaluate the resulting models

    python3 test.py

Plot the results

    python3 report.py


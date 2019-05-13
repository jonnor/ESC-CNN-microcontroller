
# Environmental Sound Classification on Microcontrollers using Convolutional Neural Networks

## Status
**Thesis ready**

## Keywords

    Wireless Sensor Networks, Embedded Systems
    Edge Computing, Edge Machine Learning
    Noise classification, Environmental Sound Classification (ESC), Urbansounds
    Tensorflow, Keras, librosa

## See also

* [Machine Learning on Embedded Systems](https://github.com/jonnor/datascience-master/tree/master/embeddedml) notes.
* [emlearn](https://github.com/jonnor/emlearn) - Machine Learning inference engine for Microcontrollers and Embedded Systems


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


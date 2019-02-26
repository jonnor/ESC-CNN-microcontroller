

## TODO

## Phase 2: Experimentation

Writing

- Outline sections for introduction
- Outline sections for methods
- Outline sections for materials
- Setup/generate Table of existing methods

Misc

- Image of overall pipeline
- Project image, title page

Urbansound8k

- Try run preprocessing on cloud
- Test SBCNN 44.1kHz, 128 mels
- Windowed validation performance is lower than without??
But test set seems OK??
Why? Make sure not too much zero-pad
- Try remove silent frames. And frames with zero padding?
- Try to convert FastGRNN to Keras and load in STM32AI
https://github.com/Microsoft/MMdnn
- Get data augmentations to work
- Shorter fields of view. Do they save cpu/mem?
* Estimate multiply-adds for existing models
* Try to run FastGRNN on Urbansound8k
- Test Pizcak

ESC-50/ESC-10

- Download dataset, setup pipeline

Run CNN model on microcontroller

- STM32AI: Unhardcode FFT size
- STM32AI: Support window overlap
- Test USB audio input
- Test measuring current with ST board

Verification methodology proposal

- Write it
- Ask Oliver/Kristian for feedback

## Done

- Ran FastGRNN example USPS dataset
- Tested DenseNet for Urbansound8k
- Sent email for info from dilated conv authors 
- Sent email for info from LD-CNN authors
- Tested multiple-instance learning for Urbansound8k
- Test Dilated CNN for Urbansound8k
- Test a SB-CNN model for Urbansound8k
- Test a trivial audio custom model with SMT32CubeAI.
First crack detection.
9000 MACC, 2 ms classifier. 8 frames, under 15 ms log-mel preprocessing.
- Test CNN performance on STM32AI. 80Mhz.
float32 1024bin FFT,30mels,32frames log-mel preprocessing under 68ms.
float32 517k MACC CNN classification 78ms. Approx 6M MACCS/second.
- Trigger LED change to reflect model predictions
- Check how the neural networks are implemented in STM32CubeAI
- Tool for extracting MACC from Keras/TensorFlow model. `./experiments/speechcommands/featurecomplexity.py`
* Tensorflow speechcommand, test to change from MFCC to mel-spec
* Run Tensorflow speechcommand examples, check perf against published
- Test standard models examples on STM32 devkits.
AudioLoop has USB Audio out, useful for recording test data.
ST BlueSensor Android app useful for testing.
Built-in example also had BT audio out (but locked at 8kHz?)
- Move project to dedicated git repo
- Setup skeleton of report Latex/Markdown
- Setup Travis CI
- Installed STM32Cube AI toolchain, and build STM32 AI examples (HAR)
- Make a shortlist of datasets to consider
- Order STM32 devkits


## Status






## TODO


Report

- Include 1D CNN methods. Table + plot of complexity vs performance
- Include results into report. Boxplot, table with average,std perf
- Check in results files into git
- Finish Urbansound8k section
- Finish Hardware platform section
- Finish Software platform section
- Move some parts into dedicated Theory section?
- Outline sections for introduction
- Outline sections for methods
- Outline sections for materials
- Finish draft methods. Ask for feedback OK


Model evaluation

- Write all settings/parameters to a file when ran
- Use best voted performance to pick model
- Flatten settings structure in train
- Allow to specify hyperparameters on cmdline
- Perform a hyperparameter search
- Merge all experiments into single .csv file
- Check windowing functions, esp last frame and padding
- Setup GPU training. Preload feature files into memory?
- Check data augmentations working
Check the generated files wrt originals
! Initially `sbcnn16k32aug` did (little bit) worse than `sbcnn16k30`.
May require different hyperparameters? Maybe need to train for much longer?


Experiments

- Try depthwise-separable SB-CNN
- Try stacked 1D convolution in front 
- Try global fully convolutional
- Try Multiple Instance Learning again?

Code quality

- Fix tests and Travis CI build
- Add end2end tests

Run CNN model on microcontroller

- Run a LD-CNN-nodelta model on microcontroller. Try training for 16kHz, 30 mels?
- STM32AI: Test different FFT/mel sizes
- STM32AI: Support window overlap?
- STM32AI: Report/fix melspec preprocessing bug
https://community.st.com/s/topic/0TO0X0000003iUqWAI/stm32-machine-learning-ai
- Test USB audio input
- Test measuring current with ST board

Reproducability

- Try run preprocessing on cloud

ESC-50/ESC-10

- Download dataset, setup pipeline

Dissemination

- Image of overall pipeline
- Project image, title page


## Done

- STM32AI. Made tool for updating window functions.
https://github.com/jonnor/emlearn/blob/master/examples/window-function.py
- Test 16k30 SB-CNN model.
No compression. 3168k MACC CNN. 200kB flash, 27kB RAM. 396-367 ms
4 bit compression. 144kB flash, 398ms. Approx 8M MACCS/second
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
Approx 4M MACCS/second.
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





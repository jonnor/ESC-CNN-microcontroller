
## Questions

Where should review of existing methods go?

What should go into methods, except for Experimental Setup?

Is anything missing from Materials section?

## TODO


Report


- Finish introduction
- Finish draft methods. Ask for feedback OK
- Finish Software platform section
- Finish Existing methods
- Include results into report. Boxplot, table with average,std perf


Model evaluation

- Write all settings/parameters to a file when ran
- Flatten settings structure in train
- Allow to specify hyperparameters on cmdline
- Use best voted performance to pick model
- Merge all experiments into single .csv file
- Check windowing functions, esp last frame and padding
- Setup GPU training


Experiments

- Try different voting overlaps
- Try stacked 1D convolution in front 

Evaluation

- Test a reduced number of classes
- Test only foreground samples

Code quality

- Fix tests and Travis CI build
- Add end2end tests

Run CNN model on microcontroller

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






Feedback needed

- CNN/NN section
- Results/Discussion/Conclusion

## TODO

### Draft 2

Materials

- Add images of compared models

Background

- ...Fill out NN/CNN section

### Draft 3

Results

- Measure runtime on device for latest models
- Use Strided-DS-24 as chosen model (confusion matrix etc), instead of auto "best"
- Make plots a bit prettier

- Include error analysis
- Finish Discussion and Conclusion
- Add picture of demo setup

Final

- Do a figure/table captions pass.
Do they explain the figure setup/contents OK?
- Do a spell checking pass
- Do a grammar checking pass. LanguageTool + Grammarly
- Ask review pass with M
- Send to OK for feedback
- Send to J. for review
- Send to M for review

### Draft 3

- Fix test/validation sets.
- Remove duplicated 24 filter model
- Plot performance of models relative to fold
- MAYBE: Profile to see what makes training slow

Abstract

- Write it!

Add Acknowledgements

- Kristian
- Oliver
- Marianna
- John

### After report


Dissemination

- Image of overall project/system
- Project image, title page
- Record a demo video
- Write a blogpost
- Publish on Arxiv? cs.LG cs.SD eess.AS stat.ML

Related

- STM32AI: Test different FFT/mel sizes
- STM32AI: Report/fix melspec preprocessing bug
https://community.st.com/s/topic/0TO0X0000003iUqWAI/stm32-machine-learning-ai
- Test USB audio input for classifying on device

Experiment

- MAYBE: Fix train and validation generators to be single-pass? 

Code quality

- Add end2end tests
- Check windowing functions, esp last frame and padding


## Done

- Investigated why MobileNets etc use much more RAM than SB-CNN.
For SB-CNN (Conv2d->MaxPooling2d), X-CUBE-AI fuses in the MaxPooling op, and reduces RAM usage by the pooling factor (4-9x).
For MobileNet this optimization breaks down, because pooling is not used.
Layer 2 is then typically too large.
Instead one can pre-scale down using strided convolutions.
When done from layer 1, this brings RAM usage under control
- Fixed CUDA issue with SB-CNN. Can run 5x train at same time with minibatch 100,
however am still CPU bound and GPU utilization only 30%. Also small batches seem to perform worse.
With 400 batches and 3 processes, GPU utilization only 20%
- Tested SystemPerformance tool on STM32.
Standalone tool works nicely, gives performance for entire network.
Interactive profiler "Validation tool" did not work, STMCubeMX fails to communicate with firmware.
Firmware seems to work fine, says "ready to receive host command".
Validation tool seems to be only tool that can give per-layer inference times. 
- Test GPU training on GTX2060.
20 seconds instead of 170 seconds per epoch on mobilenets. 8.5x speedup
1 model only utilizing 33% of GPU power. Can theoretically run multiple models in parallell, for over 20x speedup
Under 30 minutes per experiment on all 10 folds.
However SB-CNN fails with cudaNN error.
https://github.com/tensorflow/tensorflow/issues/24828
https://github.com/keras-team/keras/issues/1538
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



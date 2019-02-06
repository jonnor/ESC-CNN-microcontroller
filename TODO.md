

## TODO

## Phase 2: Experimentation


Speech Command dataset

* Verify/fix LIME explainer. Verify with trivial model on sinewaves?
Split spectrogram evenly over RGB. Test that roundtrip is correct
* Send --preprocess=logmel patch as PR to TensorFlow
* Train Keras models, compare with Tensorflow

Experiments

* Try to run FastGRNN
- Try to run 1D CNN on STM32 (AclNet LL)

Run CNN model on microcontroller

- Test a trivial audio custom model with SMT32CubeAI
** Use STM preprocessing when training
** Maybe without max normalization?
** Update the generated model
** Try enabling USB audio input
** Try playing back tests speaker->microphone
** Change to shorter classification window and FFT
- Test measuring current with ST board

Verification methodology proposal

- Write it
- Ask Oliver/Kristian for feedback

## Done

- Test CNN performance on STM32AI. 80Mhz.
float32 1024bin FFT,30mels,32frames log-mel preprocessing under 68ms.
float32 517k MACC CNN classification 78ms. Approx 6M MACCS/second.
- Trigger LED change to reflect model predictions
- Check how the neural networks are implemented in STM32CubeAI
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



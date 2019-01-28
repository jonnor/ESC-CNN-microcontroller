

## TODO

## Milestone 1: Ready for experimentation


Test & choose datasets

- Decide evaluation metrics

Methodology proposal

- Write it
- Ask Oliver/Kristian for feedback

Speech Command dataset

* Run Tensorflow examples, check perf against published
* Compare Keras models to Tensorflow
* Test to change from MFCC to mel-spec
* Visualize results with LIME

Run CNN model on microcontroller

- Test a trivial audio custom model with SMT32CubeAI
- Trigger LED change to reflect model predictions
- Check how the neural networks are implemented in STM32CubeAI
- Test measuring current


## Done

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



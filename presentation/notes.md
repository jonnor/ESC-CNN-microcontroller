
20-30 minutes.
20 slides?

6 sections a 5 minutes 

# TODO

- Go through presentation once
- Add notes to all slides


# Scope

- Problem definition
- Urbansound8k dataset
- Melspectrogram
- CNN audio model
- SB-CNN model

Out-of-scope

- Teaching basic Machine Learning
- Teaching CNN?

# Results

- Bottleneck and EffNet performed poorly
- Practical speedup not linear with MACC
- Striding on input means downsampling.
Could a smaller feature representation perform similar? 

Remarks

- GPU utilization was poor. Probably model too small

# Challenges & Solutions

RAM, CPU, FLASH

- Reduce input feature representation. mels & time.
Use from LD-CNN. 31 frames @ 22050 Hz (720 ms)
- Use CNNs instead of DNN.
Shown to get higher performance, more parameter efficient
- Reduce use of overlaps
No overlap. Existing uses maximum overlap, 100x the CPU time.
- Apply more efficient convolutional blocks
Depthwise separable, Spatially Separable, Bottleneck, EffNet
- 

## Misc

SB-CNN paper cited 280+ times






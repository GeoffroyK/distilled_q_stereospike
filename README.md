# Quantization for Stereospike
## 1. Hardware Requirements
- Depthwise convolution:
  - IN: Events (1-bit)
  - Weights: int8 [-128,127]
  - OUT: int32 --> (later we should be able to change the bitwidth)
- Pointwise convolution:
  - IN: int32 --> (later we should be able to change the bitwidth)
  - Weights: int8 [-128,127]
  - OUT: int32 --> (later we should be able to chnage the bitwidth)
- Last layers (prediction layers) also include bias --> represent bias using int32.

## 2. Pretrained Model Weights
You can download the pretrained weights of the models for both experiments (QKD, Stereospike) here:
[Download Pretrained Links](https://drive.google.com/drive/folders/1yG0c-reaNzCwuCU-2aJ4HVLOC9idG5NU)

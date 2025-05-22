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

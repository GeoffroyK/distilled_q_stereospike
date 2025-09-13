# Quantization for Stereospike
## 1. download dataset
You need to download datasets before implementing.  
train_set.pt: [Link](https://sutdapac-my.sharepoint.com/:u:/g/personal/tomomasa_yamasaki_mymail_sutd_edu_sg/EUWn0HuiTZhDiUfuKmA5lUAB8V_WG63LmvW0gEmlcqHLjQ?e=VjAXT2)  
val_set.pt: [Link](https://sutdapac-my.sharepoint.com/:u:/g/personal/tomomasa_yamasaki_mymail_sutd_edu_sg/Ee8hO7Nhng9CnCG3CWBwUjwBXnmmZXiuiYAaEUM_fNfI7w?e=xlzhfO)   
test_set.pt: [Link](https://sutdapac-my.sharepoint.com/:u:/g/personal/tomomasa_yamasaki_mymail_sutd_edu_sg/EViMoTcOFCdAui2c0zojOAoB7n9KzOulfPWDNNcZQE60AQ?e=ukQjhy)


## FYI: SQAKD
GitHub: https://github.com/kaiqi123/SQAKD


# 13th Septermber, 2025
### updated by Tomomasa Yamasaki

#### Background  
•	StereoSpike: A spiking neural network model for stereo matching.  
•	SQAKD: “Student-Quantized Adaptive Knowledge Distillation”, a method to enable quantization-aware training with knowledge distillation.  
•	Goal: Apply SQAKD to StereoSpike to reduce model size and enable efficient deployment on edge devices without losing accuracy.  

#### Current Progress  
• Replaced nn.Conv2d layers in StereoSpike with quantization-enabled modules:
```
	•	QConv (standard convolution with quantization)
	•	QConv_DW (depthwise convolution, rewritten for SQAKD)
	•	QConv_PW (pointwise convolution, rewritten for SQAKD)
```
•	Implemented teacher-student framework:  
```python
Teacher(pretrained) = SIMPLIFIED_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v2
Student(Quantizable StereoSpike) = SQAKD_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v4
```
•	Integrated QAT with SQAKD loss functions.  
•	Training script updated to support quantization-aware training (``StereosSpike_SQAKD_v101.py``).  

#### How to Run
```python
python StereosSpike_SQAKD_v101.py
```

#### Current Issue and solutions
1. Training loss becomes NAN value
Solution: 
```
SQAKD's QConv doesn't run initialzation to define weight factors.
Therefore, Turn on initialization on QConv_DW and QConv_PW
```
Coding: Please check ``network/custom_modules_v100.py``

2. weights becomes 0 or NAS value during training
Solution:
```
The output from QConv_DW and QConv_PW is too small to fire spike.
Therefore, make multiply_factor larger, such as 10 or 100.
```
Coding: Please check ``network/SNN_models_simpquant_v100.py``
```python
#SNN_models_simpquant_v100.py

class SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v4(NeuromorphicNet):
    """
    Uses separable convolutions for a lighter model

    See this excellent article to know moreabout separable convolutions:
    https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False, multiply_ratio=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]):
        super().__init__(detach_reset=detach_reset)

        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2
        self.multiply_factor = multiply_factor

        bottom_multiply_ratio = multiply_ratio[0]
        conv1_multiply_ratio = multiply_ratio[1]
        conv2_multiply_ratio = multiply_ratio[2]
        conv3_multiply_ratio = multiply_ratio[3]
        conv4_multiply_ratio = multiply_ratio[4]
        neck_multiply_ratio = multiply_ratio[5]
        deconv4_multiply_ratio = multiply_ratio[6]
        deconv3_multiply_ratio = multiply_ratio[7]
        deconv2_multiply_ratio = multiply_ratio[8]
        deconv1_multiply_ratio = multiply_ratio[9]
        pred4_multiply_ratio = multiply_ratio[10]
        pred3_multiply_ratio = multiply_ratio[11]
        pred2_multiply_ratio = multiply_ratio[12]
        pred1_multiply_ratio = multiply_ratio[13]
```
The quantized model (``SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v4``) has the parameter to change ``multply_factor``. If you want to use original multi_factor from Stereospike, set ``multiply_ratio=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]``



 



# Keras implementation of Deeplabv2 with tensorflow backend
DeepLabv2 is one of state-of-art deep learning models for semantic image segmentation.  
Based on the paper [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915v1.pdf).

__Note: For now only VGG-16 encoder is implemented.__

__Note2: Weights are stored in the repository using GitLFS. Therefore - it may take some time to clone this repo.__

Code is based on the repository [DavideA/deeplabv2-keras](https://github.com/DavideA/deeplabv2-keras), which was implemented using Theano backend.

### How to use
Execute `python testing.py` (Input image is defined in the testing.py, so edit it to use different image).

### Requirements (Tested on those versions)
Python==2.7.12
Keras==2.2.4  
tensorflow==1.9.0  
CUDA==9.0.176   

### Improvements (TODO)
* add fully-connected CRF post processing [(pydensecrf?)](https://github.com/lucasb-eyer/pydensecrf)
* add ResNet-101 encoder 

### Segmentation results example
<p align="center">
    <img src="example/example_results.png" width=600></br>
</p>

# DPRNet : Deep 3D Point based Residual Network for Semantic Segmentation and Classification of 3D Point Clouds

# Usage

This code is tested in Ubuntu 16.04 LTS with CUDA 8.0 and Tensorflow-gpu==1.4.
First of all convolutional operators needs to be compiled as follow:

cd tf_ops/conv3p/
chmod 777 tf_conv3p_compile.sh
./tf_conv3p_compile.sh -a


To train object classification, execute

python3 train_modelnet40.py [epoch]

To evaluate, execute

python3 eval_modelnet40.py [epoch]

Similar procedure is required for scene segmentation task. By default 'epoch' is 0. You can resume the training by passing epoch number in the above command.

# Training Data

Data folder contains links of the datasets for both classification and semantic segmentation task.


## Troubleshooting 

If you are using Tensorflow 1.4, you might want to try compiling with `tf_conv3p_compile_tf14.sh` instead. It fixes some include paths due to `nsync_cv.h`, and set the flag `_GLIBCXX_USE_CXX11_ABI=0` to make it compatible to libraries compiled with GCC version earlier than 5.1. 


## Dependencies

This code includes the following third party libraries and data:

- Scaled exponential linear units (SeLU) for self-normalization in neural network.

- ModelNet40 data from PointNet

- Some other utility code from Pointwise CNN

- h5py

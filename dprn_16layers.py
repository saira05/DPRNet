# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:39:39 2018

@author: saira
"""
import tensorflow as tf
import numpy as np

import util
import selu

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
conv3p_module = tf.load_op_library(BASE_DIR + '/tf_ops/conv3p/tf_conv3p.so')

def conv3p(points_tensor, input_tensor, kernel_tensor, stride, voxel_size):
    return conv3p_module.conv3p(points_tensor, input_tensor, kernel_tensor, stride, voxel_size);

@tf.RegisterGradient('Conv3p')
def _conv3p_grad(op, grad_from_next_layer):
    """The derivatives for convolution.
    Args:
        op: the convolution op.
        grad_from_next_layer: the tensor representing the gradient w.r.t. the output
    Returns:
        the gradients w.r.t. the point tensor, input tensor, and the filter
    """
    points = op.inputs[0]
    input = op.inputs[1]
    filter = op.inputs[2]
    stride = op.inputs[3]
    voxel_size = op.inputs[4]

    input_grad, filter_grad = conv3p_module.conv3p_grad(grad_from_next_layer, points, input, filter, stride, voxel_size)
    return [None, input_grad, filter_grad, None, None]

class PointConvNet:
    def __init__(self, num_class):
        self.num_class = num_class

    def model(self, points_tensor, input_tensor, is_training=True):
        """
        Arguments:
            points_tensor: [b, n, 3] point cloud
            input_tensor: [b, n, channels] extra data defined for each point
        """
        b = points_tensor.get_shape()[0].value
        n = points_tensor.get_shape()[1].value
        in_channels = input_tensor.get_shape()[2].value


        voxel_size = tf.constant([0.1])
        stride = tf.constant([1, 1, 1])
        filter1_tensor = tf.get_variable("filter1", [3, 3, 3, in_channels, 9])
        conv1 = conv3p(points_tensor, input_tensor, filter1_tensor, stride, voxel_size);
        relu1 = selu.selu(conv1)

        stride = tf.constant([2, 2, 2])
        filter2_tensor = tf.get_variable("filter2", [3, 3, 3, 9, 9])
        conv2 = conv3p(points_tensor, relu1, filter2_tensor, stride, voxel_size);
        relu2 = selu.selu(conv2)

        stride = tf.constant([3, 3, 3])
        filter3_tensor = tf.get_variable("filter3", [3, 3, 3, 9, 9])
        conv3 = conv3p(points_tensor, relu2, filter3_tensor, stride, voxel_size);
        skipCon1 = tf.add(relu1, conv3)
        relu3 = selu.selu(skipCon1)

        stride = tf.constant([4, 4, 4])
        filter4_tensor = tf.get_variable("filter4", [3, 3, 3, 9, 9])
        conv4 = conv3p(points_tensor, relu3, filter4_tensor, stride, voxel_size);
        relu4 = selu.selu(conv4)

        stride = tf.constant([5, 5, 5])
        filter5_tensor = tf.get_variable("filter5", [3, 3, 3, 9, 9])
        conv5 = conv3p(points_tensor, relu4, filter5_tensor, stride, voxel_size);
        skipCon2 = tf.add(relu3, conv5)
        relu5 = selu.selu(skipCon2)

        stride = tf.constant([6, 6, 6])
        filter6_tensor = tf.get_variable("filter6", [3, 3, 3, 9, 9])
        conv6 = conv3p(points_tensor, relu5, filter6_tensor, stride, voxel_size);
        relu6 = selu.selu(conv6)

        stride = tf.constant([7, 7, 7])
        filter7_tensor = tf.get_variable("filter7", [3, 3, 3, 9, 9])
        conv7 = conv3p(points_tensor, relu6, filter7_tensor, stride, voxel_size);
        skipCon3 = tf.add(relu5, conv7, name = None)
        relu7 = selu.selu(skipCon3)
        
        stride = tf.constant([8, 8, 8])
        filter8_tensor = tf.get_variable("filter8", [3, 3, 3, 9, 9])
        conv8 = conv3p(points_tensor, relu7, filter8_tensor, stride, voxel_size);
        relu8 = selu.selu(conv8)

        stride = tf.constant([9, 9, 9])
        filter9_tensor = tf.get_variable("filter9", [3, 3, 3, 9, 9])
        conv9 = conv3p(points_tensor, relu8, filter9_tensor, stride, voxel_size);
        skipCon4 = tf.add(relu7, conv9)
        relu9 = selu.selu(skipCon4)

        stride = tf.constant([10, 10, 10])
        filter10_tensor = tf.get_variable("filter10", [3, 3, 3, 9, 9])
        conv10 = conv3p(points_tensor, relu9, filter10_tensor, stride, voxel_size);
        relu10 = selu.selu(conv10)

        stride = tf.constant([11, 11, 11])
        filter11_tensor = tf.get_variable("filter11", [3, 3, 3, 9, 9])
        conv11 = conv3p(points_tensor, relu10, filter11_tensor, stride, voxel_size);
        skipCon5 = tf.add(relu9, conv11)
        relu11 = selu.selu(skipCon5)

        stride = tf.constant([12, 12, 12])
        filter12_tensor = tf.get_variable("filter12", [3, 3, 3, 9, 9])
        conv12 = conv3p(points_tensor, relu11, filter12_tensor, stride, voxel_size);
        relu12 = selu.selu(conv12)

        stride = tf.constant([13, 13, 13])
        filter13_tensor = tf.get_variable("filter13", [3, 3, 3, 9, 9])
        conv13 = conv3p(points_tensor, relu12, filter13_tensor, stride, voxel_size);
        skipCon6 = tf.add(relu11, conv13,name = None)
        relu13 = selu.selu(skipCon6)

        stride = tf.constant([14, 14, 14])
        filter14_tensor = tf.get_variable("filter14", [3, 3, 3, 9, 9])
        conv14 = conv3p(points_tensor, relu13, filter14_tensor, stride, voxel_size);
        relu14 = selu.selu(conv14)

        stride = tf.constant([15, 15, 15])
        filter15_tensor = tf.get_variable("filter15", [3, 3, 3, 9, 9])
        conv15 = conv3p(points_tensor, relu14, filter15_tensor, stride, voxel_size);
        skipCon7 = tf.add(relu13, conv15)
        relu15 = selu.selu(skipCon7)

        stride = tf.constant([16, 16, 16])
        filter16_tensor = tf.get_variable("filter16", [3, 3, 3, 9, 9])
        conv16 = conv3p(points_tensor, relu15, filter16_tensor, stride, voxel_size);
        relu16 = selu.selu(conv16)




        feat = tf.concat([relu1, relu2, relu3, relu4, relu5, relu6, relu7, relu8, relu9, relu10, relu11, relu12, relu13, relu14, relu15, relu16], axis=2) #local features
        view = tf.reshape(feat, [-1, n * 144])                  #global feature
        
        fc1 = tf.contrib.layers.fully_connected(view, 512 , activation_fn=selu.selu)
        
        dropout1 = selu.dropout_selu(x=fc1, rate=0.5, training=is_training)

        fc2 = tf.contrib.layers.fully_connected(dropout1, self.num_class, activation_fn=selu.selu)

        return fc2

    def loss(self, logits, labels):
        """
        Arguments:
            logits: prediction with shape [batch_size, num_class]
            labels: ground truth scalar labels with shape [batch_size]
        """
        #onehot_labels = tf.one_hot(labels, depth=self.num_class)        
        #e = tf.losses.softmax_cross_entropy(onehot_labels, logits)
        
        e = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        e = tf.reduce_mean(e)
        return e

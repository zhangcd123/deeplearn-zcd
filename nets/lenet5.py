"""
Decripition:lenet5 implement by python used tensorflow 1.12.0 for recongnize hand written
Input:32x32
Layers:7 layers include:
    conv1(k:5x5,f:6,s:1)    28x28  out = (in-k)/s+1
    pool1(k:2x2,f:6:s:2)    14x14  out = (in-k)/s+1
    conv2(k:5x5,f:16,s:1)   10x10
    pool2(k:2x2,f:16,s:2)   5x5
    conv3(k:5x5,f:120,s:1)  1x1
    fc(84)  84
    output: 10
Paper:Gradient-Based Learning Applied to Document Recognition
Author:Cara
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
maxpool = tf.contrib.layers.max_pool2d


def build_net(inputs,scope):
    """
    buile lenet5 
    """
    #28x28 conv
    net = slim.conv2d(inputs
                        , 6
                        , [5,5]
                        , stride=1
                        , activation_fn=None
                        , normalizer_fn=None
                        , scope=scope+'_conv1')
    
    #14x14 pool
    net = maxpool(net
                    , kernel_size=2
                    , stride=2
                    , scope=scope+'_max_pool1')

    #10x10 conv
    net = slim.conv2d(net
                        , 16
                        , [5,5]
                        , stride=1
                        , activation_fn=None
                        , normalizer_fn=None
                        , scope=scope+'_conv2')
    #5x5 pool
    net = maxpool(net
                    , kernel_size=2
                    , stride=2
                    , scope=scope+'_max_pool2')
    
    #1x1 conv
    net = slim.conv2d(net
                        , 120
                        , [5,5]
                        , stride=1
                        , activation_fn=None
                        , normalizer_fn=None
                        , scope=scope+'_conv3')

    #fc
    net = slim.fully_connected(net
                                , 84
                                , scope='fc/fc_1')
    #fc
    logit = slim.fully_connected(net
                                , 10
                                , scope='fc/fc_2')
    return logit
    


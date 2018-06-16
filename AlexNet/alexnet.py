"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np
import scipy.io

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        #1st Layer: Conv (w ReLu)
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        #2nd Layer: Norm 
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        #3rd Layer: Pool
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        #4th Layer: Conv (w ReLu)
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        #5th Layer: Norm
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        #6th Layer: Pool
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        #7th Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
        #8th Layer: Conv (w ReLu)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        #9th Layer: Conv (w ReLu)
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        #10th Layer: Pool
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        #11th Layer: FC (w ReLu) 
        fc6 = fc(tf.reshape(pool5, [-1, 6*6*256]), 6*6*256, 4096, relu=True, name='fc6')
        #12th Layer: Dropout
        dropout6 = dropout(fc6, self.KEEP_PROB)
        #13th Layer: FC (w ReLu)
        fc7 = fc(dropout6, 4096, 4096, relu=True, name='fc7')
        #14th Layer: Dropout
        dropout7 = dropout(fc7, self.KEEP_PROB)
        #15th Layer: FC (w Softmax)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
        """Different weights:
        weights_dict2 = scipy.io.loadmat("imagenet-matconvnet-alex.mat")
        weights_dict2 = weights_dict2['params'][0]
        weights_dict['conv1'][0] = weights_dict2[0][1]
        weights_dict['conv1'][1] = weights_dict2[1][1].flatten()
        weights_dict['conv2'][0] = weights_dict2[2][1]
        weights_dict['conv2'][1] = weights_dict2[3][1].flatten()
        weights_dict['conv3'][0] = weights_dict2[4][1]
        weights_dict['conv3'][1] = weights_dict2[5][1].flatten()
        weights_dict['conv4'][0] = weights_dict2[6][1]
        weights_dict['conv4'][1] = weights_dict2[7][1].flatten()
        weights_dict['conv5'][0] = weights_dict2[8][1]
        weights_dict['conv5'][1] = weights_dict2[9][1].flatten()
        weights_dict['fc6'][0] = weights_dict2[10][1].reshape(-1, weights_dict['fc6'][0].shape[-1])
        weights_dict['fc6'][1] = weights_dict2[11][1].flatten()
        weights_dict['fc7'][0] = weights_dict2[12][1][0][0]
        weights_dict['fc7'][1] = weights_dict2[13][1].flatten()
        weights_dict['fc8'][0] = weights_dict2[14][1][0][0]
        weights_dict['fc8'][1] = weights_dict2[15][1].flatten()
        """
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, relu,name):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        relu = tf.nn.softmax(act)
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

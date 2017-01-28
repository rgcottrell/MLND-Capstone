# ==============================================================================
# Copyright 2017 Robert Cottrell. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

def variable_summaries(var):
    """Attach summaries to a tensor for TensorBoard visualization."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
    """Create a weight variable of the requested shape."""
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
        variable_summaries(weights)
    return weights

def bias_variable(shape):
    """Create a bias variable of the requested shape."""
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.constant(0.01, shape=shape))
        variable_summaries(biases)
    return biases
    
def reshape(name, x, shape):
    """Reshape a tensor."""
    return tf.reshape(x, shape, name=name)

def convolution(layer_name, x, kernel_size, in_depth, out_depth, strides=1, padding='SAME'):
    """Create a 2D convolution layer."""
    with tf.name_scope(layer_name):
        weights = weight_variable([kernel_size, kernel_size, in_depth, out_depth])
        biases = bias_variable([out_depth])
        with tf.name_scope('convolution'):
            preactivate = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding=padding)
            preactivate = tf.nn.bias_add(preactivate, biases)
            tf.summary.histogram('pre_activations', preactivate)
        activations = tf.nn.elu(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations

def max_pool(layer_name, x, ksize, strides, padding='SAME'):
    """Create a max pooling layer."""
    pooled = tf.nn.max_pool(x, [1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding, name=layer_name)
    tf.summary.histogram(layer_name + 'pooling', pooled)
    return pooled
    
def local_response(layer_name, x):
    """Create a local response normalization layer."""
    lrn = tf.nn.local_response_normalization(x, name=layer_name)
    tf.summary.histogram(layer_name + '_normalizations', lrn)
    return lrn
    
def dropout(layer_name, x, keep_prob):
    """Create a dropout layer with the requested keep probability."""
    with tf.name_scope(layer_name):
        dropped = tf.nn.dropout(x, keep_prob)
        tf.summary.histogram('dropouts', dropped)
    return dropped

def convolution_block(layer_name, x, in_depth, out_depth, strides, dropout_keep_prob):
    """Create a block of layers for each convolution stage."""
    with tf.name_scope(layer_name):
        conv = convolution('conv', x, 5, in_depth, out_depth)
        conv = max_pool('pool', conv, 2, strides)
        conv = local_response('lrn', conv)
        conv = dropout('dropout', conv, dropout_keep_prob)
    return conv    

def fully_connected(layer_name, x, in_width, out_width):
    """Create a fully connected layer."""
    with tf.name_scope(layer_name):
        weights = weight_variable([in_width, out_width])
        biases = bias_variable([out_width])
        with tf.name_scope('fully_connected'):
            preactivate = tf.add(tf.matmul(x, weights), biases)
            tf.summary.histogram('pre_activations', preactivate)
        activations = tf.nn.elu(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations
    
def fully_connected_block(layer_name, x, in_width, out_width, dropout_keep_prob):
    """Create a fully connected layer followed by dropout."""
    with tf.name_scope(layer_name):
        fc = fully_connected('fc', x, in_width, out_width)
        fc = dropout('dropout', fc, dropout_keep_prob)
    return fc
 
def classifier(layer_name, x, in_depth, out_depth):
    """Create a classiier readout layer."""
    with tf.name_scope(layer_name):
        weights = weight_variable([in_depth, out_depth])
        biases = bias_variable([out_depth])
        with tf.name_scope('classifier'):
            readout = tf.add(tf.matmul(x, weights), biases)
            tf.summary.histogram('readout', readout)
    return readout    

def inference(x, dropout_keep_prob):
    """Build the multi-digit recognition classifier model."""
    image = reshape('input_reshape', x, [-1, 54, 54, 3])
    tf.summary.image('input_images', image, 10)

    conv1 = convolution_block('conv1', image, 3, 48, 2, dropout_keep_prob)
    conv2 = convolution_block('conv2', conv1, 48, 64, 1, dropout_keep_prob)
    conv3 = convolution_block('conv3', conv2, 64, 128, 2, dropout_keep_prob)
    conv4 = convolution_block('conv4', conv3, 128, 160, 1, dropout_keep_prob)
    conv5 = convolution_block('conv5', conv4, 160, 192, 2, dropout_keep_prob)
    conv6 = convolution_block('conv6', conv5, 192, 192, 1, dropout_keep_prob)
    conv7 = convolution_block('conv7', conv6, 192, 192, 2, dropout_keep_prob)
    conv8 = convolution_block('conv8', conv7, 192, 192, 1, dropout_keep_prob)
    flatten = reshape('flatten_reshape', conv8, [-1, 4*4*192])
    fc9 = fully_connected_block('fc9', flatten, 4*4*192, 3072, dropout_keep_prob)
    fc10 = fully_connected_block('fc10', fc9, 3072, 3072, dropout_keep_prob)
    
    logits_array = [
        classifier('y0', fc10, 3072, 7),
        classifier('y1', fc10, 3072, 10,),
        classifier('y2', fc10, 3072, 10),
        classifier('y3', fc10, 3072, 10),
        classifier('y4', fc10, 3072, 10),
        classifier('y5', fc10, 3072, 10),    
    ]
    return logits_array

def cross_entropy_loss(name, logits, labels):
    """Calculate the cross entropy loss for a single classifier."""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar(name, loss)
    return loss

def loss(num_digits, logits_array, labels_array):
    """Calculates the loss from the logits and labels."""
    losses = [
        cross_entropy_loss('xentropy_y0', logits_array[0], labels_array[0]),
        cross_entropy_loss('xentropy_y1', logits_array[1], labels_array[1]),
        cross_entropy_loss('xentropy_y2', logits_array[2], labels_array[2]),
        cross_entropy_loss('xentropy_y3', logits_array[3], labels_array[3]),
        cross_entropy_loss('xentropy_y4', logits_array[4], labels_array[4]),
        cross_entropy_loss('xentropy_y5', logits_array[5], labels_array[5])
    ]
    cases = [
        (tf.equal(num_digits, 1), lambda:
            losses[0] + losses[1]),
        (tf.equal(num_digits, 2), lambda:
            losses[0] + losses[1] + losses[2]),
        (tf.equal(num_digits, 3), lambda:
            losses[0] + losses[1] + losses[2] + losses[3]),
        (tf.equal(num_digits, 4), lambda:
            losses[0] + losses[1] + losses[2] + losses[3] + losses[4]),
        (tf.equal(num_digits, 5), lambda:
            losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5])
    ]
    loss = tf.case(cases, lambda: losses[0], exclusive=True)
    tf.summary.scalar('total_loss', loss)
    return loss    

def accuracy(num_digits, logits_array, labels_array):
    """Calculates the accuracy of the batch."""
    correct_predictions = [
        tf.equal(tf.argmax(logits_array[0], 1), tf.cast(labels_array[0], tf.int64)),
        tf.equal(tf.argmax(logits_array[1], 1), tf.cast(labels_array[1], tf.int64)),
        tf.equal(tf.argmax(logits_array[2], 1), tf.cast(labels_array[2], tf.int64)),
        tf.equal(tf.argmax(logits_array[3], 1), tf.cast(labels_array[3], tf.int64)),
        tf.equal(tf.argmax(logits_array[4], 1), tf.cast(labels_array[4], tf.int64)),
        tf.equal(tf.argmax(logits_array[5], 1), tf.cast(labels_array[5], tf.int64))
    ]
    correct_prediction = correct_predictions[0]
    correct_prediction = tf.where(num_digits > 0, tf.logical_and(correct_prediction, correct_predictions[1]), correct_prediction)
    correct_prediction = tf.where(num_digits > 1, tf.logical_and(correct_prediction, correct_predictions[2]), correct_prediction)
    correct_prediction = tf.where(num_digits > 2, tf.logical_and(correct_prediction, correct_predictions[3]), correct_prediction)
    correct_prediction = tf.where(num_digits > 3, tf.logical_and(correct_prediction, correct_predictions[4]), correct_prediction)
    correct_prediction = tf.where(num_digits > 4, tf.logical_and(correct_prediction, correct_predictions[5]), correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def predict(logits_array):
    """Returns the log probability predictions"""
    log_probs = [
        tf.nn.log_softmax(logits_array[0], name='y0_pred'),
        tf.nn.log_softmax(logits_array[1], name='y1_pred'),
        tf.nn.log_softmax(logits_array[2], name='y2_pred'),
        tf.nn.log_softmax(logits_array[3], name='y3_pred'),
        tf.nn.log_softmax(logits_array[4], name='y4_pred'),
        tf.nn.log_softmax(logits_array[5], name='y5_pred')
    ]   
    return log_probs 
    
def training(loss, global_step):
    """Set up the training ops."""
    return tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)

#!/usr/bin/env python

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

import input
import model

import numpy as np
import tensorflow as tf
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'logs', 'Directory to write event logs and checkpoints.')
tf.app.flags.DEFINE_integer('max_steps', 0, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of samples per batch.')
tf.app.flags.DEFINE_boolean('clean', False, 'Delete log directory before starting.')

class NumDigitsHook(tf.train.SessionRunHook):
    """Pick digits in next batch based on the inverse of the decaying accuracy."""
    def __init__(self, num_digits, accuracy):
        self._num_digits = num_digits
        self._accuracy = accuracy

    def begin(self):
        self._weights = np.zeros(5)

    def before_run(self, run_context):
        # Choose the next digit length proportional to the inverse of the
        # decayed accuracies. Clip the maximum value for small accuracies.
        weights = np.minimum(1000., 1. / self._softmax(self._weights))
        weights /= np.sum(weights)
        num_digits = np.random.choice(5, p=weights) + 1
        return tf.train.SessionRunArgs([self._num_digits, self._accuracy], feed_dict={self._num_digits: num_digits})

    def after_run(self, run_context, run_values):
        num_digits, accuracy = run_values.results
        self._weights[num_digits-1] *= 0.9
        self._weights[num_digits-1] += accuracy

    def _softmax(self, values):
        # Compute softmax using normalized values for numeric stability.
        e_values = np.exp(values - np.max(values))
        return e_values / np.sum(e_values)

class LoggerHook(tf.train.SessionRunHook):
    """Display loss and accuracy for training batches."""
    def __init__(self, num_digits, loss, accuracy):
        self._num_digits = num_digits
        self._loss = loss
        self._accuracy = accuracy
    
    def begin(self):
        self._step = -1
    
    def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs([self._num_digits, self._loss, self._accuracy])
        
    def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        num_digits, loss, accuracy = run_values.results
        format_str =  ('%s: step %d, digits: %d, accuracy: %.1f%%, loss: %.4f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), self._step, num_digits, 100. * accuracy, loss, examples_per_sec, sec_per_batch))

def run_training():
    """Create model graph and start training."""
    with tf.Graph().as_default():
        # Create a vartiable to count the number of times the optimizer has
        # run. This equals the number of batches processed.
        global_step = tf.Variable(0, trainable=False, name='global_step')
        
        # The number of digits the current batch is training.
        num_digits = tf.placeholder(tf.int32, name='num_digits')
        
        # Create images and labels pipeline.
        images, labels_array = input.training_input(num_digits, FLAGS.batch_size)

         # Build a Graph that computes predictions from the inference model.
        logits_array = model.inference(images, tf.constant(0.5))
        
        # Calculate the loss and accuracy.
        loss = model.loss(num_digits, logits_array, labels_array)
        accuracy = model.accuracy(num_digits, logits_array, labels_array)
        
        # Build a Graph that trains the modelwith one batch of examples and
        # updates the model parameters.
        train_op = model.training(loss, global_step)
        
        hooks = [
            NumDigitsHook(num_digits, accuracy),
            LoggerHook(num_digits, loss, accuracy),
            tf.train.NanTensorHook(loss)
        ]
        if FLAGS.max_steps > 0:
            hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.max_steps))
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.logdir, hooks=hooks) as sess:
            while not sess.should_stop():
                sess.run([train_op])
        
def main(argv=None):
    if FLAGS.clean and tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
    run_training()

if __name__ == '__main__':
    tf.app.run()
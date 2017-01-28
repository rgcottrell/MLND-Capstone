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

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'logs', 'Directory to restore checkpoints.')
tf.app.flags.DEFINE_string('set', 'validation', 'The data set to evaluate.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of samples per batch.')

def score(predictions, labels):
    """Calculate the number of correct predictions in the batch."""    
    correct = 0
    coverage = 0
    total = len(predictions[0])
    for index in range(total):
        p0 = np.argmax(predictions[0][index])
        p1 = np.argmax(predictions[1][index])
        p2 = np.argmax(predictions[2][index])
        p3 = np.argmax(predictions[3][index])
        p4 = np.argmax(predictions[4][index])
        p5 = np.argmax(predictions[5][index])
        
        l0 = labels[0][index]
        l1 = labels[1][index]
        l2 = labels[2][index]
        l3 = labels[3][index]
        l4 = labels[4][index]
        l5 = labels[5][index]
                
        # Check whether the prediction was correct. There is no partial credit
        # for getting the parts of the prediction right. The counter and all
        # valid digits must agree.
        result = True
        if l0 != p0:
            result = False
        if l0 > 0 and l1 != p1:
            result = False
        if l0 > 1 and l2 != p2:
            result = False
        if l0 > 2 and l3 != p3:
            result = False
        if l0 > 3 and l4 != p4:
            result = False
        if l0 > 4 and l5 != p5:
            result = False
        if result:
            correct += 1

        pct = 0
        if l0 == 0:
            pct = np.exp(predictions[0][index][l0])
        elif l0 == 1:
            pct = np.exp(predictions[0][index][l0] + predictions[1][index][l1])
        elif l0 == 2:
             pct = np.exp(predictions[0][index][l0] + predictions[1][index][l1] + predictions[2][index][l2])
        elif l0 == 3:
            pct = np.exp(predictions[0][index][l0] + predictions[1][index][l1] + predictions[2][index][l2] + predictions[3][index][l3])
        elif l0 == 4:
            pct = np.exp(predictions[0][index][l0] + predictions[1][index][l1] + predictions[2][index][l2] + predictions[3][index][l3] + predictions[4][index][l4])
        elif l0 == 5:
            pct = np.exp(predictions[0][index][l0] + predictions[1][index][l1] + predictions[2][index][l2] + predictions[3][index][l3] + predictions[4][index][l4] + predictions[5][index][l5])
        elif l0 == 6:
             pct = np.exp(predictions[0][index][l0])
        if pct >= 0.98:
            coverage += 1
        
    return correct, coverage, total
    
def run_evaluation():
    """Run an evaluation loop on the data set to compute accuracy."""
    with tf.Graph().as_default():
        # Create images and labels pipeline.
        images, labels = input.evaluation_input(FLAGS.set, FLAGS.batch_size)
        
         # Build a Graph that computes predictions from the inference model.
        logits = model.inference(images, 1.0)
        
        # The Op to return predictions.
        predict = model.predict(logits)
        
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        # Create a saver to restore the model from the latest checkpoint.
        saver = tf.train.Saver()
                            
        with tf.Session() as sess:
            # Initialize the session.
            sess.run(init_op)
            
            # Restore the latest model snapshot.
            ckpt = tf.train.get_checkpoint_state(FLAGS.logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored: {}'.format(ckpt.model_checkpoint_path))
            else:
                print('No checkpoint file found!')
                return
                
            # Start the queue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            # Loop through evaluation data to calculate accuracy.
            total_correct = 0
            total_coverage = 0
            total_samples = 0
            try:
                while not coord.should_stop():
                    run_ops = [
                        predict[0], predict[1], predict[2], predict[3], predict[4], predict[5],
                        labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]
                    ]
                    p0, p1, p2, p3, p4, p5, y0, y1, y2, y3, y4, y5 = sess.run(run_ops)
                    correct, coverage, total = score([p0, p1, p2, p3, p4, p5], [y0, y1, y2, y3, y4, y5])
                    total_correct += correct
                    total_coverage += coverage
                    total_samples += total
                    print('Accuracy: %d/%d = %.2f%%, Coverage: %d/%d - %.2f%%' % (correct, total, 100. * correct / total, coverage, total, 100. * coverage / total))
            except tf.errors.OutOfRangeError:
                print('Finished processing data.')
                coord.request_stop()
            
            # Print final accuracy count
            print('Total Accuracy: %d/%d = %.2f%%,  Coverage: %d/%d - %.2f%%' % (total_correct, total_samples, 100. * total_correct / total_samples,  total_coverage, total_samples, 100. * total_coverage / total_samples))
            print('Coverage at 98%:')
            
            # Wait for threads
            coord.join(threads)

def main(argv=None):
    run_evaluation()

if __name__ == "__main__":
    tf.app.run()
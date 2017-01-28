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

import os
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'logs', 'Directory to restore checkpoints.')
tf.app.flags.DEFINE_string('predictdir', 'predict/', 'Directory with images to predict labels for')
    
def run_predictions():
    """Predict labels for new images."""
    with tf.Graph().as_default():
        # Create images pipeline.
        path = tf.placeholder(tf.string)
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, 54, 54)
        image = tf.image.per_image_standardization(image)
        
         # Build a Graph that computes predictions from the inference model.
        logits = model.inference(image, 1.0)
        
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
            
            with open('predict.csv', 'w') as csv:
                # Loop through files in the predict directory.
                csv.write('image,label,label_pct,counter,counter_pct,y1,y1_pct,y2,y2_pct,y3,y3_pct,y4,y4_pct,y5,y5_pct\n')
                for filename in os.listdir(FLAGS.predictdir):
                    file = FLAGS.predictdir + filename
                    run_ops = [
                        predict[0], predict[1], predict[2], predict[3], predict[4], predict[5]
                    ]
                    p0, p1, p2, p3, p4, p5 = sess.run(run_ops, feed_dict={path: file})
                    i0 = np.argmax(p0)
                    i1 = np.argmax(p1)
                    i2 = np.argmax(p2)
                    i3 = np.argmax(p3)
                    i4 = np.argmax(p4)
                    i5 = np.argmax(p5)
                
                    if i0 == 0:
                        label = 'X'
                        label_pct = np.exp(p0[0][i0])
                    if i0 == 1:
                        label = '{}'.format(p1[i1])
                        label_pct = np.exp(p0[0][i0] + p1[0][i1])
                    if i0 == 2:
                        label = '{}{}'.format(i1, i2)
                        label_pct = np.exp(p0[0][i0] + p1[0][i1] + p2[0][i2])
                    if i0 == 3:
                        label = '{}{}{}'.format(i1, i2, i3)
                        label_pct = np.exp(p0[0][i0] + p1[0][i1] + p2[0][i2] + p3[0][i3])
                    if i0 == 4:
                        label = '{}{}{}{}'.format(i1, i2, i3, i4)
                        label_pct = np.exp(p0[0][i0] + p1[0][i1] + p2[0][i2] + p3[0][i3] + p4[0][i4])
                    if i0 == 5:
                        label = '{}{}{}{}{}'.format(i1, i2, i3, i4, i5)
                        label_pct = np.exp(p0[0][i0] + p1[0][i1] + p2[0][i2] + p3[0][i3] + p4[0][i4] + p5[0][i5])
                    if i0 == 6:
                        label = '+'
                        label_pct = np.exp(p0[0][i0])
                
                    csv.write(','.join([
                        file,
                        label, '%0.1f' % (100 * label_pct),
                        str(i0), '%0.1f' % (100 * np.exp(p0[0][i0])),
                        str(i1), '%0.1f' % (100 * np.exp(p1[0][i1])),
                        str(i2), '%0.1f' % (100 * np.exp(p2[0][i2])),
                        str(i3), '%0.1f' % (100 * np.exp(p3[0][i3])),
                        str(i4), '%0.1f' % (100 * np.exp(p4[0][i4])),
                        str(i5), '%0.1f' % (100 * np.exp(p5[0][i5])),
                    ]))
                    csv.write('\n')

def main(argv=None):
    run_predictions()

if __name__ == "__main__":
    tf.app.run()
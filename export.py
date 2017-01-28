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

import model

import tensorflow as tf
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'logs', 'Directory to write event logs and checkpoints.')

def export_model():
    """Freeze and export model."""
    with tf.Graph().as_default() as graph:
        # Create images and labels pipeline.
        images = tf.placeholder(tf.float32, name='x_input')
        
         # Build a Graph that computes predictions from the inference model.
        logits = model.inference(images, tf.constant(1.0))
        
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
                step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Model restored: {}'.format(ckpt.model_checkpoint_path))
            else:
                print('No checkpoint file found!')
                return

            # Export variables as constant.
            out_graph = graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), ['y0_pred', 'y1_pred', 'y2_pred', 'y3_pred', 'y4_pred', 'y5_pred'])
            
            # Save the model.
            with tf.gfile.GFile(FLAGS.logdir + '/model.pb', 'wb') as file:
                file.write(out_graph.SerializeToString())
            print('%d ops in the final graph.' % len(out_graph.node))

def main(argv=None):
    export_model()

if __name__ == "__main__":
    tf.app.run()
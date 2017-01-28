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

def load_training_data(filename, batch_size):
    """Load the queues for training images and labels."""
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[""], [0], [0], [0], [0], [0], [0]]
    path, y0, y1, y2, y3, y4, y5 = tf.decode_csv(value, record_defaults=record_defaults)
    
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.random_crop(image, [54, 54, 3])
    image = tf.image.per_image_standardization(image)
    
    images, ys0, ys1, ys2, ys3, ys4, ys5 = tf.train.shuffle_batch(
        [image, y0, y1, y2, y3, y4, y5], batch_size=batch_size, num_threads=1,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
    
    return images, [ys0, ys1, ys2, ys3, ys4, ys5]
    
def load_evaluation_data(filename, batch_size):
    """Load the queues for evaluation images and labels."""
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[""], [0], [0], [0], [0], [0], [0]]
    path, y0, y1, y2, y3, y4, y5 = tf.decode_csv(value, record_defaults=record_defaults)
    
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 54, 54)
    image = tf.image.per_image_standardization(image)
    
    images, ys0, ys1, ys2, ys3, ys4, ys5 = tf.train.batch(
        [image, y0, y1, y2, y3, y4, y5], batch_size=batch_size,
        capacity=3 * batch_size, allow_smaller_final_batch=True)
    
    return images, [ys0, ys1, ys2, ys3, ys4, ys5]

def merge_inputs(num_digits, t1, t2, t3, t4, t5):
    """Select from one of the inputs based on number of digits."""
    cases = [
        (tf.equal(num_digits, 1), lambda: t1),
        (tf.equal(num_digits, 2), lambda: t2),
        (tf.equal(num_digits, 3), lambda: t3),
        (tf.equal(num_digits, 4), lambda: t4),
        (tf.equal(num_digits, 5), lambda: t5)
    ]
    return tf.case(cases, lambda: t1, exclusive=True)

def training_input(num_digits, batch_size):
    """Create input pipeline for training data."""
    images1, labels1 = load_training_data('train-1.csv', batch_size)
    images2, labels2 = load_training_data('train-2.csv', batch_size)
    images3, labels3 = load_training_data('train-3.csv', batch_size)
    images4, labels4 = load_training_data('train-4.csv', batch_size)
    images5, labels5 = load_training_data('train-5.csv', batch_size)
    
    merged_images = merge_inputs(num_digits, images1, images2, images3, images4, images5)
    merged_labels = [
        merge_inputs(num_digits, labels1[0], labels2[0], labels3[0], labels4[0], labels5[0]),
        merge_inputs(num_digits, labels1[1], labels2[1], labels3[1], labels4[1], labels5[1]),
        merge_inputs(num_digits, labels1[2], labels2[2], labels3[2], labels4[2], labels5[2]),
        merge_inputs(num_digits, labels1[3], labels2[3], labels3[3], labels4[3], labels5[3]),
        merge_inputs(num_digits, labels1[4], labels2[4], labels3[4], labels4[4], labels5[4]),
        merge_inputs(num_digits, labels1[5], labels2[5], labels3[5], labels4[5], labels5[5]),
    ]
    return merged_images, merged_labels
    
def evaluation_input(set, batch_size):
    """Create input pipeline for evaluation data."""
    return load_evaluation_data(set + '.csv', batch_size)
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

import h5py
from PIL import Image

def decode_string(ds):
    return ''.join(chr(x[0]) for x in ds)
    
def decode_label(data, label_ds):
    label = ''
    if label_ds.shape[0] == 1:
        label = str(int(label_ds[0][0]) % 10)
    else:
        for label_array in label_ds:
            label += str(int(data[label_array[0]][0][0]) % 10)
    return label

def decode_coordinates(data, coordinate_ds):
    coordinates = []
    if coordinate_ds.shape[0] == 1:
        coordinates.append(int(coordinate_ds[0][0]))
    else:
        for coordinate_array in coordinate_ds:
            coordinates.append(int(data[coordinate_array[0]][0][0]))
    return coordinates

def process_image(in_filename, out_filename, left, top, right, bottom):
    with Image.open(in_filename) as image:
        image = image.crop((left, top, right, bottom))
        image = image.resize((64, 64), Image.LANCZOS)
        image.save(out_filename)

def merge_bbox(lefts, tops, widths, heights):
    merged_left = None
    merged_top = None
    merged_right = None
    merged_bottom = None
    for index in range(len(lefts)):
        left = lefts[index]
        top = tops[index]
        right = left + widths[index]
        bottom = top + heights[index]
        merged_left = left if merged_left is None else min(merged_left, left)
        merged_top = top if merged_top is None else min(merged_top, top)
        merged_right = right if merged_right is None else max(merged_right, right)
        merged_bottom = bottom if merged_bottom is None else max(merged_bottom, bottom)

    horizontal_padding = int(0.15 * (merged_right - merged_left))
    vertical_padding = int(0.15 * (merged_bottom - merged_top))
    merged_left -= horizontal_padding
    merged_right += horizontal_padding
    merged_top -= vertical_padding
    merged_bottom += vertical_padding
    return merged_left, merged_top, merged_right, merged_bottom

def process_dataset(data_dir):
    with h5py.File(data_dir + 'digitStruct.mat', 'r') as data:
        names_ds = data['digitStruct/name']
        bboxes_ds = data['digitStruct/bbox']

        mappings = ['Image,Label']
        for names_array, bboxes_array in zip(names_ds, bboxes_ds):
            bbox_group = data[bboxes_array[0]]
            label = decode_label(data, bbox_group['label'])
            lefts = decode_coordinates(data, bbox_group['left'])
            tops = decode_coordinates(data, bbox_group['top'])
            widths = decode_coordinates(data, bbox_group['width'])
            heights = decode_coordinates(data, bbox_group['height'])
            left, top, right, bottom = merge_bbox(lefts, tops, widths, heights)

            name = decode_string(data[names_array[0]])
            in_filename = data_dir + name
            out_filename = data_dir + 'processed-' + name
            process_image(in_filename, out_filename, left, top, right, bottom)

            mappings.append(name + ',' + label)

        with open(data_dir + 'mappings.csv', 'w') as file:
            for mapping in mappings:
                file.write(mapping + '\n')
                
def main():
    process_dataset('train/')
    process_dataset('test/')
    process_dataset('extra/')
    
if __name__ == "__main__":
    main()
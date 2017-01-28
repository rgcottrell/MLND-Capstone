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

def main():
    # digits = [0,0,0,0,0,0]
    # with open('train/mappings.csv') as file:
    #     file.readline()
    #     for line in file:
    #         name, label = line.strip().split(',')
    #         digits[len(label)-1] += 1
    #     with open('extra/mappings.csv') as file:
    #         file.readline()
    #         for line in file:
    #             name, label = line.strip().split(',')
    #             digits[len(label)-1] += 1
    # print digits
    
    digits = [0,0,0,0,0,0,0,0,0,0]
    with open('train/mappings.csv') as file:
        file.readline()
        for line in file:
            name, label = line.strip().split(',')
            for x in label:
                digits[int(x)] += 1
            
    with open('extra/mappings.csv') as file:
        file.readline()
        for line in file:
            name, label = line.strip().split(',')
            for x in label:
                digits[int(x)] += 1
    print digits
    
if __name__ == "__main__":
    main()
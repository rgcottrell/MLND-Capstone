# MLND-Capstone

Capstone project for Udacity's Machine Learning Engineer Nanodegree

## Problem Statement

This project will use a deep learning convolutional neural network to classify multi-digit number sequences.

The project is a TensorFlow implementation of "Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks", available [here](http://arxiv.org/pdf/1312.6082.pdf)

## Dataset

The dataset can be downloaded from [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)

Download train.tar.gz, test.tar.gz, and extra.tar.gz and extract into this folder.

## Scripts

Several Python scripts are available to train the model:

* ```preprocess.py```: Use this script to preprocess each of the downloaded images. This will detect the bounding boxes around the house numbers, crop out the numbers, and resize the numbers to 64x64 images.
* ```split.py```: Use this script to split the data into training, validation, and test sets.
* ```train.py```: Use this script to train the model. Training with an NVIDIA Titan X (Pascal) GPU will take approximately three days to reach 95% accuracy.
* ```eval.py```: Use this script to periodically evaluate the validation set during training. After finishing training, run with ```-set=test``` to calculate accuracy and coverage on the test set.
* ```predict.py```: Use this script to make predictions on new images in the ```predict``` directory. Images should be 64x64 jpegs. Results will be available in ```predict.csv```.

Two Python utility modules are used for training and evaluation.

* ```model.py```: This defines the model, optimizer, and loss functions.
* ```input.py```: This defines the image input and preprocessing data pipelines.

### Report

A final report explaining this project and the surrounding problem domain is available as ```report.pdf```.

### License

The code for this project is open source and available under the terms of the license in this repository.
#!/bin/bash
./mnist_filter --input data/mnist/train-images-idx3-ubyte \
               --filter sobel \
               --output results/filtered_images/
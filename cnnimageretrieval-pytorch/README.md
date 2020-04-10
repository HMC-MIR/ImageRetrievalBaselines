# Sheet ID CNNS

This repository contains all the code for interfacing with CNNs used in the paper "Sheet Music Identification Using Measure-Based CNN Features" by Hubinger, Khant, Kurashige, Amin, and Tsai.

## Installation

To install this repository, simply

1. `git clone https://github.com/evhub/cnnimageretrieval-pytorch.git` and
2. `make install`.

## Running Benchmarks

To run the benchmark systems on the Sheet ID dataset, you will first need to install [`score-retrieval`](https://github.com/evhub/score-retrieval). Once that is done, simply `make run-benchmarks` and all the benchmark results will be deposited in the folder `./logs`.

## Fine-Tuning

To run fine-tuning on the Sheet ID dataset, you will first need to install [`score-retrieval`](https://github.com/evhub/score-retrieval). Then, either `make train-vgg-gem` to fine-tune from scratch or copy existing fine-tuning weights into `./weights`.

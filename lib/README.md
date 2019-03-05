# PyTorch libraries for Attraction Field Learning 

## Introduction
### 1. PyTorch extension for AFM Transform

We extend the pytorch for transforming line segments to attraction field map. To use this operator, please make sure a nvidia GPU card is equipped and avaliable for computing.

### 2. Squeeze module

TODO: extend the cython implementation to pytorch CPP/CUDA extension

## Build

Run the following command in this directory
```
make
```

## Usage
### 1. AFM transform operator

This operator can transform a set of line segments into attraction field map and support the batch processing. 
- Input parameters@lines:
A pytorch CUDA float tensor with shape of Nx4
- Input parameters@shape_info:
A pytorch CUDA int tensor with shape of Bx4, where B is the batch_size.



#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load MNIST dataset
Tensor* load_mnist_images(const char* filename);

// Load MNIST labels 
Tensor* load_mnist_labels(const char* filename);

#ifdef __cplusplus
}
#endif



#endif
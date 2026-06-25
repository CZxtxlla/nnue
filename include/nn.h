#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "ops.h"
#include <math.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Tensor* weight; // shape [in_features, out_features]
    Tensor* bias; // shape [1, out_features]
} LinearLayer;


// stuff for linear layer
LinearLayer* create_linear_layer(int in_features, int out_features, DeviceType device); // xavier uniform weights and bias
Tensor* linear_forward(LinearLayer* layer, Tensor* input); // pass the input tensor through linear layer
void free_linear_layer(LinearLayer* layer); // free the memory allocated for the linear layer and its tensors

// stuff for MLP
typedef struct {
    LinearLayer** layers; // all the layers in the mlp
    int num_layers;
    DeviceType dev;
} MLP;


MLP* create_mlp(int* architecture, int num_layers, DeviceType device); // create mlp given the layer sizes and number of layers
Tensor* mlp_forward(MLP* model, Tensor* input); // perform forward pass through the mlp
Tensor** mlp_get_parameters(MLP* model, int* out_num_parameters); // get array with all learnable parameters (weights and biases)
void free_mlp(MLP* model); // free memory allocated for the mlp and its layers
int save_mlp(MLP* model, char* location); // save MLP to specified file 
MLP* load_mlp(char* location, DeviceType device); // load MLP saved in location to model


#ifdef __cplusplus
}
#endif

#endif
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

// ----- NNUE architecture ------

typedef struct {
    LinearLayer* feature_transformer; // sparse layer 768 -> 128
    LinearLayer** hidden_layers; // dense layers
    int num_hidden_layers;
    DeviceType dev;
} NNUE;

// in features 768, accumulator size 128, hidden dims {256, 32, 1}
NNUE* create_nnue(int in_features, int accumulator_size, int* hidden_dims, int num_hidden_layers);

// input tensors contain the active feature indices
Tensor* nnue_forward(NNUE* model, Tensor* white_inputs, Tensor* black_inputs, Tensor* side_to_move);
Tensor* nnue_forward_nonsparse(NNUE* model, Tensor* white_inputs, Tensor* black_inputs, Tensor* side_to_move);

Tensor** nnue_get_parameters(NNUE* model, int* out_num_parameters);
void free_nnue(NNUE* model);
int save_nnue(NNUE* model, const char* location);
NNUE* load_nnue(char* location, DeviceType device);
int save_nnue_quantized(NNUE* model, const char* location);

#define FEATURE_SIZE 768 
#define L1_SIZE 128



#ifdef __cplusplus
}
#endif

#endif
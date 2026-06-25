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

typedef struct {
    int in_channels; // input depth i.e. for rgb image, 3 (red, green blue), for grey image 1
    int out_channels; // number of filters
    int filter_size; // 3 for a 3x3 filter
    int stride; // how far filter moves after calculating each pixel
    int padding; // width of zero-padding pixels

    Tensor* weight; // shape [out_channels, in_channels, filter_size, filter_size]
    Tensor* bias; // shape [1, out_channels, 1, 1]
} Conv2dLayer;

typedef struct {
    int filter_size;
    int padding;
    int stride;
} MaxPool2DLayer;

// stuff for linear layer
LinearLayer* create_linear_layer(int in_features, int out_features, DeviceType device); // xavier uniform weights and bias
Tensor* linear_forward(LinearLayer* layer, Tensor* input); // pass the input tensor through linear layer
void free_linear_layer(LinearLayer* layer); // free the memory allocated for the linear layer and its tensors

//stuff for convolutional layer
Conv2dLayer* create_conv2d_layer(int in_channels, int out_channels, int filter_size, int stride, int padding, DeviceType device);
Tensor* conv2d_forward(Conv2dLayer* layer, Tensor* input);
void free_conv2d_layer(Conv2dLayer* layer);

// stuff for max pooling layer
MaxPool2DLayer* create_maxpool2d_layer(int filter_size, int stride, int padding);
Tensor* maxpool2d_layer_forward(MaxPool2DLayer* layer, Tensor* input);
void free_maxpool2d_layer(MaxPool2DLayer* layer);

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


// stuff for CNN
typedef struct {
    Conv2dLayer* conv1;
    MaxPool2DLayer* pool1;
    Conv2dLayer* conv2;
    MaxPool2DLayer* pool2;
    LinearLayer* fc1;
    LinearLayer* fc2;
} SimpleCNN;

SimpleCNN* create_simple_cnn(DeviceType device);
Tensor* cnn_forward(SimpleCNN* model, Tensor* input);
Tensor** cnn_get_parameters(SimpleCNN* model, int* out_num_parameters);
void free_simple_cnn(SimpleCNN* model);



#ifdef __cplusplus
}
#endif

#endif
#include "../include/nn.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int cudaFree(void* devPtr);


// helper for xavier uniform initialization
static float random_float(float limit) {
    float unit = (float)rand() / (float)RAND_MAX;
    return (unit * 2.0f - 1.0f) * limit;
}


// ------------ Linear Layer -----------------

LinearLayer* create_linear_layer(int in_features, int out_features, DeviceType device) {
    // create layer with xavier uniform initialization for weights and bias
    LinearLayer* layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    if (layer == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for linear layer.\n");
        return NULL;
    }

    int weight_shape[] = {in_features, out_features};
    int bias_shape[] = {1, out_features};

    layer->weight = create_tensor(weight_shape, 2, DEVICE_CPU, true);
    if (layer->weight == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for linear layer.\n");
        free(layer);
        return NULL;
    }

    layer->bias = create_tensor(bias_shape, 2, DEVICE_CPU, true);
    if (layer->bias == NULL) {
        fprintf(stderr, "Error: problem creating bias tensor for linear layer.\n");
        free(layer);
        return NULL;
    }

    // Xavier uniform initialization for weights and biases
    
    float limit = sqrtf(6.0f / (float)(in_features + out_features));
    for (int i = 0; i < layer->weight->size; i++) {
        layer->weight->cpu_data[i] = random_float(limit);
    }
    for (int i = 0; i < layer->bias->size; i++) {
        layer->bias->cpu_data[i] = 0.01f;
    }
    if (device == DEVICE_GPU) {
        tensor_to_device(layer->weight, DEVICE_GPU);
        tensor_to_device(layer->bias, DEVICE_GPU);
    }

    return layer;

}

Tensor* linear_forward(LinearLayer* layer, Tensor* input) {
    // perform the forward pass through the linear layer and return the output tensor
    Tensor* output = tensor_matmul(input, layer->weight); // output = input @ weight

    Tensor* result = tensor_add_bias(output, layer->bias); // result = output + bias
    
    // keep the intermediate node alive for use in backward
    return result;
}

void free_linear_layer(LinearLayer* layer) {
    if (layer != NULL) {
        free_tensor(layer->weight);
        free_tensor(layer->bias);
        free(layer);
    }
}


// ----------- Convolutional layer ---------------

Conv2dLayer* create_conv2d_layer(int in_channels, int out_channels, int filter_size, int stride, int padding, DeviceType device) {
    // create convolutional layer given input parameters

    Conv2dLayer* layer = (Conv2dLayer*)malloc(sizeof(Conv2dLayer));
    if (layer == NULL) {
        fprintf(stderr, "Error: problem with allocating memory for convolutional layer initialization.\n");
        return NULL;
    }
    
    // initialize parameters
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->filter_size = filter_size;
    layer->stride = stride;
    layer->padding = padding;

    // [number of filters, channels per filter, filter_height, filter_width]
    int weight_shape[] = {out_channels, in_channels, filter_size, filter_size};

    int bias_shape[] = {1, out_channels, 1, 1};

    layer->weight = create_tensor(weight_shape, 4, DEVICE_CPU, true);
    if (layer->weight == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for convolutional layer weights.\n");
        free(layer);
        return NULL;
    }

    layer->bias = create_tensor(bias_shape, 4, DEVICE_CPU, true);
    if (layer->bias == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for convolutional layer bias.\n");
        free(layer);
        return NULL;
    }

    // xavier uniform initialization

    int filter_area = filter_size * filter_size;
    float filter_in = filter_area * in_channels; // in_features
    float filter_out = filter_area * out_channels; // out_features
    float limit = sqrtf(6.0f / (float) (filter_in + filter_out));

    for (int i = 0; i < layer->weight->size; i++) {
        layer->weight->cpu_data[i] = random_float(limit);
    }
    for (int i = 0; i < layer->bias->size; i++) {
        layer->bias->cpu_data[i] = 0.01f;
    }

    if (device == DEVICE_GPU) {
        tensor_to_device(layer->weight, DEVICE_GPU);
        tensor_to_device(layer->bias, DEVICE_GPU);
    }

    return layer;
}


void free_conv2d_layer(Conv2dLayer* layer) {
    // free all memory allocated for the convolutional layer
    if (layer != NULL) {
        free_tensor(layer->weight);
        free_tensor(layer->bias);
        free(layer);
    }
}

Tensor* conv2d_forward(Conv2dLayer* layer, Tensor* input) {
    // pass to the dispatcher which will do all the linking for the computations graph for backprop.
    return tensor_conv2d(input, layer->weight, layer->bias, layer->stride, layer->padding);
}

// -------- Max pooling layer ----------

MaxPool2DLayer* create_maxpool2d_layer(int filter_size, int stride, int padding) {
    MaxPool2DLayer* layer = (MaxPool2DLayer*)malloc(sizeof(MaxPool2DLayer));
    if (layer == NULL) {
        fprintf(stderr, "Error: problem with allocating memory for max pooling layer initialization.\n");
        return NULL;
    }
    layer->filter_size = filter_size;
    layer->stride = stride;
    layer->padding = padding;
    return layer;
}

Tensor* maxpool2d_layer_forward(MaxPool2DLayer* layer, Tensor* input) {
    return maxpool2d_forward(input, layer->filter_size, layer->stride, layer->padding);
}

void free_maxpool2d_layer(MaxPool2DLayer* layer) {
    if (layer != NULL) {
        free(layer);
    }
}

// --------- MLP ----------

MLP* create_mlp(int* architecture, int num_layers, DeviceType device) {
    MLP* model = (MLP*)malloc(sizeof(MLP));
    if (model == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for the MLP struct.\n");
        return NULL;
    }
    model->dev = device;
    model->num_layers = num_layers - 1;
    model->layers = (LinearLayer**)malloc(model->num_layers * sizeof(LinearLayer*));
    if (model->layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP layers array.\n");
        free(model);
        return NULL;
    }

    for (int i = 0; i < model->num_layers; i++) {
        model->layers[i] = create_linear_layer(architecture[i], architecture[i + 1], device);
    }
    return model;
}

Tensor* mlp_forward(MLP* model, Tensor* input) {
    Tensor* current = input;
    for (int i = 0; i < model->num_layers; i++) {
        current = linear_forward(model->layers[i], current);
        // apply relu to hidden layers
        if (i < model->num_layers - 1) {
            current = tensor_relu(current);
        }
    }

    return current;
}   

Tensor** mlp_get_parameters(MLP* model, int* out_num_parameters) {
    *out_num_parameters = model->num_layers * 2; // each layer has weight and bias
    Tensor** params = (Tensor**)malloc(*out_num_parameters * sizeof(Tensor*));
    if (params == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP parameters array.\n");
        return NULL;
    }

    for (int i = 0; i < model->num_layers; i++) {
        params[i * 2] = model->layers[i]->weight;
        params[i * 2 + 1] = model->layers[i]->bias;
    }
    return params;
}

void free_mlp(MLP* model) {
    if (model != NULL) {
        for (int i = 0; i < model->num_layers; i++) {
            free_linear_layer(model->layers[i]);
        }
        free(model->layers);
        free(model);
    }
}

#define MLP_MAGIC_NUMBER 0x534D4C50 // magic number to identify MLP file

int save_mlp(MLP* model, char* location) {
    if (model == NULL || location == NULL) {
        return 0; // failure
    } 
    FILE* file = fopen(location, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file %s.\n", location);
        return 0;
    }

    uint32_t magic = MLP_MAGIC_NUMBER;
    fwrite(&magic, sizeof(uint32_t), 1, file); // write the identifier

    fwrite(&(model->num_layers), sizeof(int), 1, file); // write the number of layers

    for (int i = 0; i < model->num_layers; i++) {
        LinearLayer* layer = model->layers[i];

        int in_features = layer->weight->shape[0];
        int out_features = layer->weight->shape[1];

        // write the dimensions of the layer
        fwrite(&in_features, sizeof(int), 1, file);
        fwrite(&out_features, sizeof(int), 1, file);

        // get cpu version of the data
        float* weight = (float*)malloc(layer->weight->size * sizeof(float));
        float* bias = (float*)malloc(layer->bias->size * sizeof(float));

        tensor_download_data(layer->weight, weight);
        tensor_download_data(layer->bias, bias);

        fwrite(weight, sizeof(float), layer->weight->size, file);
        fwrite(bias, sizeof(float), layer->bias->size, file);

        free(weight);
        free(bias);
    }

    fclose(file);
    return 1; // success
}

MLP* load_mlp(char* location, DeviceType device) {
    if (location == NULL) {
        return NULL; // failure
    } 
    FILE* file = fopen(location, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file %s.\n", location);
        return NULL;
    }

    // verify magic number
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 || magic != MLP_MAGIC_NUMBER) {
        fprintf(stderr, "Error: problem reading file or wrong file type.\n");
        fclose(file);
        return NULL;
    }

    int num_layers_in_file;
    if (fread(&num_layers_in_file, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: problem reading file.\n");
        fclose(file);
        return NULL;
    }
    // create MLP
    MLP* model = (MLP*)malloc(sizeof(MLP));
    if (model == NULL) {
        fprintf(stderr, "Error: memory allocation failed for MLP.\n");
        fclose(file);
        return NULL;
    }

    model->num_layers = num_layers_in_file;
    model->dev = device;

    model->layers = (LinearLayer**)calloc(num_layers_in_file, sizeof(LinearLayer*));

    for (int i = 0; i < num_layers_in_file; i++) {
        int in_features, out_features;

        if (fread(&in_features, sizeof(int), 1, file) != 1 || fread(&out_features, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "Error: missing layer dimensions.\n");
            fclose(file);
            free_mlp(model);
            return NULL;
        }

        model->layers[i] = create_linear_layer(in_features, out_features, DEVICE_CPU);

        LinearLayer* layer = model->layers[i];
        fread(layer->weight->cpu_data, sizeof(float), layer->weight->size, file);
        fread(layer->bias->cpu_data, sizeof(float), layer->bias->size, file);
        if (device == DEVICE_GPU) {
            tensor_to_device(layer->weight, DEVICE_GPU);
            tensor_to_device(layer->bias, DEVICE_GPU);
        }
    }
    fclose(file);
    return model;
}

// ------------ CNN ------------------

Tensor* tensor_flatten(Tensor* input) {
    int batch_size = input->shape[0];
    int flat_features = input->shape[1] * input->shape[2] * input->shape[3];

    int new_shape[] = {batch_size, flat_features};
    Tensor* out = create_tensor(new_shape, 2, input->device, input->requires_grad);
    if (out == NULL) {
        fprintf(stderr, "Error: problem creating output tensor in tensor_flatten.\n");
        return NULL;
    }

    out->op = OP_NONE;
    out->op = OP_FLATTEN;
    out->is_view = true;

    out->num_parents = 1;
    out->parents = (Tensor**)malloc(sizeof(Tensor*));
    if (out->parents == NULL) {
        fprintf(stderr, "Error: problem allocating memory for parents array in tensor_flatten.\n");
        free_tensor(out);
        return NULL;
    }
    out->parents[0] = input;

    if (input->device == DEVICE_CPU) {
        free(out->cpu_data);
        out->cpu_data = input->cpu_data;
        cudaFree(out->gpu_data);
        out->gpu_data = NULL;
    } else if (input->device == DEVICE_GPU) {
        free(out->cpu_data);
        out->cpu_data = NULL;
        cudaFree(out->gpu_data);
        out->gpu_data = input->gpu_data;
    }

    return out;

}

SimpleCNN* create_simple_cnn(DeviceType device) {
    // create CNN for training on MNIST
    SimpleCNN* model = (SimpleCNN*)malloc(sizeof(SimpleCNN));
    if (model == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for simpel CNN.\n");
        return NULL;
    }
    // 1 in_channel (MNIST greyscale), 8 filters, 3x3 filter, stride 1, padding 1
    model->conv1 = create_conv2d_layer(1, 8, 3, 1, 1, device);
    model->pool1 = create_maxpool2d_layer(2, 2, 0); // 2x2 filter, stride of 2

    // 8 in_channels, 16 filters, 3x3 filter, stride 1, padding 1
    model->conv2 = create_conv2d_layer(8, 16, 3, 1, 1, device);
    model->pool2 = create_maxpool2d_layer(2, 2, 0); // 2x2 filter, stride of 2

    // flattened size is 16 channels * 7 height * 7 width = 784 features
    // note: height and width are 7 cause we started out with 28 x 28 image but maxpooled twice, 
    // dividing both dimensions by 4 (since our filter is 2x2 with stride 2)
    model->fc1 = create_linear_layer(784, 128, device);
    model->fc2 = create_linear_layer(128, 10, device);

    return model;
}

Tensor* cnn_forward(SimpleCNN* model, Tensor* input) {
    Tensor* x;

    x = conv2d_forward(model->conv1, input);
    x = tensor_relu(x);
    x = maxpool2d_layer_forward(model->pool1, x);

    x = conv2d_forward(model->conv2, x);
    x = tensor_relu(x);
    x = maxpool2d_layer_forward(model->pool2, x);

    x = tensor_flatten(x);

    x = linear_forward(model->fc1, x);
    x = tensor_relu(x);
    x = linear_forward(model->fc2, x);

    return x;
}
Tensor** cnn_get_parameters(SimpleCNN* model, int* out_num_parameters) {
    // get all learnable parameters
    *out_num_parameters = 8; // 2 from conv1, 2 from conv2, 2 from fc1, 2 from fc2
    Tensor** params = (Tensor**)malloc(*out_num_parameters * sizeof(Tensor*));
    if (params == NULL) {
        fprintf(stderr, "Error: problem allocating memory for parameters array.\n");
        return NULL;
    }

    params[0] = model->conv1->weight;
    params[1] = model->conv1->bias;
    params[2] = model->conv2->weight;
    params[3] = model->conv2->bias;
    params[4] = model->fc1->weight;
    params[5] = model->fc1->bias;
    params[6] = model->fc2->weight;
    params[7] = model->fc2->bias;

    return params;
}


void free_simple_cnn(SimpleCNN* model) {
    if (model != NULL) {
        free_conv2d_layer(model->conv1);
        free_conv2d_layer(model->conv2);
        free_maxpool2d_layer(model->pool1);
        free_maxpool2d_layer(model->pool2);
        free_linear_layer(model->fc1);
        free_linear_layer(model->fc2);
        
        free(model);
    }
}
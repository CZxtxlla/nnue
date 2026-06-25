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
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

    layer->weight = create_tensor(weight_shape, 2, DEVICE_CPU, true, 0);
    if (layer->weight == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for linear layer.\n");
        free(layer);
        return NULL;
    }

    layer->bias = create_tensor(bias_shape, 2, DEVICE_CPU, true, 0);
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

// ------- NNUE ----------

NNUE* create_nnue(int in_features, int accumulator_size, int* hidden_dims, int num_hidden_layers) {
    NNUE* model = (NNUE*)malloc(sizeof(NNUE));
    if (model == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for the NNUE struct.\n");
        return NULL;
    }

    model->dev = DEVICE_GPU;
    model->num_hidden_layers = num_hidden_layers;

    if (hidden_dims[0] != accumulator_size * 2) {
        fprintf(stderr, "Error: hidden_dims[0] must be exactly twice the accumulator_size (expected %d, got %d).\n", accumulator_size * 2, hidden_dims[0]);
        free(model);
        return NULL;
    }

    model->feature_transformer = create_linear_layer(in_features, accumulator_size, DEVICE_GPU);

    model->hidden_layers = (LinearLayer**)malloc(sizeof(LinearLayer*) * num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++) {
        model->hidden_layers[i] = create_linear_layer(hidden_dims[i], hidden_dims[i + 1], DEVICE_GPU);
    }

    return model;
}

Tensor* nnue_forward(NNUE* model, Tensor* white_inputs, Tensor* black_inputs, Tensor* side_to_move) {

    Tensor* white_acc = tensor_sparse_linear_forward(white_inputs, model->feature_transformer->weight, model->feature_transformer->bias);
    Tensor* black_acc = tensor_sparse_linear_forward(black_inputs, model->feature_transformer->weight, model->feature_transformer->bias);

    Tensor* white_clipped = tensor_clipped_leaky_relu(white_acc);
    Tensor* black_clipped = tensor_clipped_leaky_relu(black_acc);

    Tensor* current_input = tensor_perspective_concat_forward(white_clipped, black_clipped, side_to_move);

    // hidden layers
    for (int i = 0; i < model->num_hidden_layers; i++) {
        Tensor* linear_out = linear_forward(model->hidden_layers[i], current_input);

        if (i < model->num_hidden_layers - 1) {
            current_input = tensor_clipped_leaky_relu(linear_out);
        } else {
            current_input = linear_out;
        }
    }

    return current_input;
}

Tensor** nnue_get_parameters(NNUE* model, int* out_num_parameters) {
    *out_num_parameters = 2 + (model->num_hidden_layers * 2);

    Tensor** params = (Tensor**)malloc(*out_num_parameters * sizeof(Tensor*));
    if (params == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for NNUE parameters array.\n");
        return NULL;
    }

    params[0] = model->feature_transformer->weight;
    params[1] = model->feature_transformer->bias;

    for (int i = 0; i < model->num_hidden_layers; i++) {
        params[2 + (i * 2)] = model->hidden_layers[i]->weight;
        params[2 + (i * 2) + 1] = model->hidden_layers[i]->bias;
    }

    return params;
}


void free_nnue(NNUE* model) {
    if (model != NULL) {
        for (int i = 0; i < model->num_hidden_layers; i++) {
            free_linear_layer(model->hidden_layers[i]);
        }
        free(model->hidden_layers);
        free_linear_layer(model->feature_transformer);
        free(model);
    }
}

#define NNUE_MAGIC_NUMBER 0x4E4E5545 // Hex for "NNUE"

int save_nnue(NNUE* model, const char* location) {
    if (model == NULL || location == NULL) {
        return 0; // failure
    } 
    FILE* file = fopen(location, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file %s.\n", location);
        return 0;
    }

    uint32_t magic = NNUE_MAGIC_NUMBER;
    fwrite(&magic, sizeof(uint32_t), 1, file); // write the identifier

    fwrite(&(model->num_hidden_layers), sizeof(int), 1, file); // write the number of hidden layers

    // Helper macro to save linear layer
    #define SAVE_LAYER(layer) do { \
        int in_feat = layer->weight->shape[0]; \
        int out_feat = layer->weight->shape[1]; \
        fwrite(&in_feat, sizeof(int), 1, file); \
        fwrite(&out_feat, sizeof(int), 1, file); \
        float* w_cpu = (float*)malloc(layer->weight->size * sizeof(float)); \
        float* b_cpu = (float*)malloc(layer->bias->size * sizeof(float)); \
        tensor_download_data(layer->weight, w_cpu); \
        tensor_download_data(layer->bias, b_cpu); \
        fwrite(w_cpu, sizeof(float), layer->weight->size, file); \
        fwrite(b_cpu, sizeof(float), layer->bias->size, file); \
        free(w_cpu); \
        free(b_cpu); \
    } while(0)

    // 3. Save the sparse feature transformer
    SAVE_LAYER(model->feature_transformer);

    // 4. Save the dense hidden layers
    for (int i = 0; i < model->num_hidden_layers; i++) {
        SAVE_LAYER(model->hidden_layers[i]);
    }

    #undef SAVE_LAYER

    fclose(file);
    return 1; // success
}

NNUE* load_nnue(char* location, DeviceType device) {
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
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 || magic != NNUE_MAGIC_NUMBER) {
        fprintf(stderr, "Error: problem reading file or wrong file type.\n");
        fclose(file);
        return NULL;
    }

    int num_hidden_in_file;
    if (fread(&num_hidden_in_file, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: problem reading NNUE file.\n");
        fclose(file);
        return NULL;
    }
    // create NNUE
    NNUE* model = (NNUE*)malloc(sizeof(NNUE));
    if (model == NULL) {
        fprintf(stderr, "Error: memory allocation failed for NNUE.\n");
        fclose(file);
        return NULL;
    }

    model->num_hidden_layers = num_hidden_in_file;
    model->dev = device;

    // helper macro to load linear layer
    #define LOAD_LAYER(target_ptr) do { \
        int in_feat, out_feat; \
        if (fread(&in_feat, sizeof(int), 1, file) != 1 || fread(&out_feat, sizeof(int), 1, file) != 1) { \
            fprintf(stderr, "Error: missing layer dimensions.\n"); \
            fclose(file); \
            free_nnue(model); \
            return NULL; \
        } \
        target_ptr = create_linear_layer(in_feat, out_feat, DEVICE_CPU); \
        fread(target_ptr->weight->cpu_data, sizeof(float), target_ptr->weight->size, file); \
        fread(target_ptr->bias->cpu_data, sizeof(float), target_ptr->bias->size, file); \
        if (device == DEVICE_GPU) { \
            tensor_to_device(target_ptr->weight, DEVICE_GPU); \
            tensor_to_device(target_ptr->bias, DEVICE_GPU); \
        } \
    } while(0)

    LOAD_LAYER(model->feature_transformer);

    model->hidden_layers = (LinearLayer**)calloc(num_hidden_in_file, sizeof(LinearLayer*));
    for (int i = 0; i < num_hidden_in_file; i++) {
        LOAD_LAYER(model->hidden_layers[i]);
    }

    #undef LOAD_LAYER

    fclose(file);
    return model;
}

#define QA 255.0 // input layer quantization
#define QB 64.0 // hidden layer quantization

// quantized
int save_nnue_quantized(NNUE* model, const char* location) {
    if (model == NULL || location == NULL) {
        return 0; // failure
    } 
    FILE* file = fopen(location, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file %s.\n", location);
        return 0;
    }

    uint32_t magic = NNUE_MAGIC_NUMBER;
    fwrite(&magic, sizeof(uint32_t), 1, file); // write the identifier

    fwrite(&(model->num_hidden_layers), sizeof(int), 1, file); // write the number of hidden layers

    // Helper macro to save linear layer
    #define SAVE_LAYER(layer, quant_weight, quant_bias) do { \
        int in_feat = layer->weight->shape[0]; \
        int out_feat = layer->weight->shape[1]; \
        fwrite(&in_feat, sizeof(int), 1, file); \
        fwrite(&out_feat, sizeof(int), 1, file); \
        float* w_cpu = (float*)malloc(layer->weight->size * sizeof(float)); \
        float* b_cpu = (float*)malloc(layer->bias->size * sizeof(float)); \
        tensor_download_data(layer->weight, w_cpu); \
        tensor_download_data(layer->bias, b_cpu); \
        \
        int32_t* w_quantized = (int32_t*)malloc(layer->weight->size * sizeof(int32_t)); \
        int32_t* b_quantized = (int32_t*)malloc(layer->bias->size * sizeof(int32_t)); \
        \
        for (size_t j = 0; j < layer->weight->size; j++) { \
            float scaled_w = roundf(w_cpu[j] * quant_weight); \
            if (scaled_w > INT32_MAX) scaled_w = INT32_MAX; \
            if (scaled_w < INT32_MIN) scaled_w = INT32_MIN; \
            w_quantized[j] = (int32_t)scaled_w; \
        } \
        \
        /* Scale and round biases */ \
        for (size_t j = 0; j < layer->bias->size; j++) { \
            float scaled_b = roundf(b_cpu[j] * quant_bias); \
            if (scaled_b > INT32_MAX) scaled_b = INT32_MAX; \
            if (scaled_b < INT32_MIN) scaled_b = INT32_MIN; \
            b_quantized[j] = (int32_t)scaled_b; \
        } \
        \
        fwrite(w_quantized, sizeof(int32_t), layer->weight->size, file); \
        fwrite(b_quantized, sizeof(int32_t), layer->bias->size, file); \
        \
        free(w_cpu); \
        free(b_cpu); \
        free(w_quantized); \
        free(b_quantized); \
    } while(0)

    SAVE_LAYER(model->feature_transformer, QA, QA);

    for (int i = 0; i < model->num_hidden_layers; i++) {
        SAVE_LAYER(model->hidden_layers[i], QB, QA * QB);
    }

    #undef SAVE_LAYER

    fclose(file);
    return 1; // success
}
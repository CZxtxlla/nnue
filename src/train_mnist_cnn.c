#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include "../include/dataset.h"

// --- Helper to find the index of the highest probability ---
static int argmax(const float* values, int row_index, int columns) {
    int best_index = 0;
    float best_value = values[row_index * columns];
    for (int col = 1; col < columns; col++) {
        float current = values[row_index * columns + col];
        if (current > best_value) {
            best_value = current;
            best_index = col;
        }
    }
    return best_index;
}

// --- The Evaluation Loop ---
void evaluate_model(SimpleCNN* model, Tensor* test_images, Tensor* test_labels) {
    int sample_count = test_images->shape[0];
    int num_classes = test_labels->shape[1];
    int batch_size = 128;
    int num_batches = sample_count / batch_size;
    
    float total_loss = 0.0f;
    int correct_predictions = 0;
    int total_evaluated = num_batches * batch_size;
    
    printf("Evaluating model on %d test samples...\n", total_evaluated);

    for (int b = 0; b < num_batches; b++) {
        int start_row = b * batch_size;
        
        Tensor* batch_inputs = tensor_slice_view(test_images, start_row, batch_size);
        Tensor* batch_labels = tensor_slice_view(test_labels, start_row, batch_size);
        
        // --- THE RESHAPE TRICK: Convert flat [128, 784] into 4D [128, 1, 28, 28] ---
        batch_inputs->ndims = 4;
        free(batch_inputs->shape);
        batch_inputs->shape = (int*)malloc(4 * sizeof(int));
        batch_inputs->shape[0] = batch_size;
        batch_inputs->shape[1] = 1;  // 1 Channel (Grayscale)
        batch_inputs->shape[2] = 28; // Height
        batch_inputs->shape[3] = 28; // Width
        
        // forward pass
        Tensor* predictions = cnn_forward(model, batch_inputs);
        Tensor* loss = tensor_cross_entropy(predictions, batch_labels);
        
        total_loss += tensor_scalar_value(loss);
        
        float* host_preds = (float*)malloc(predictions->size * sizeof(float));
        float* host_labels = (float*)malloc(batch_labels->size * sizeof(float));
        
        tensor_download_data(predictions, host_preds);
        tensor_download_data(batch_labels, host_labels);
        
        for (int i = 0; i < batch_size; i++) {
            int pred_class = argmax(host_preds, i, num_classes);
            int true_class = argmax(host_labels, i, num_classes);
            if (pred_class == true_class) {
                correct_predictions++;
            }
        }
        
        free(host_preds);
        free(host_labels);
        free_graph(loss);
        free_tensor(batch_inputs);
        free_tensor(batch_labels);
    }
    
    float accuracy = ((float)correct_predictions / total_evaluated) * 100.0f;
    printf("\n--- Test Results ---\n");
    printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct_predictions, total_evaluated);
    printf("Average Loss: %.4f\n--------------------\n\n", total_loss / num_batches);
}

// --- The Training Loop ---
SimpleCNN* run_simple_training(DeviceType device, const char* label, Tensor* images, Tensor* labels) {
    int sample_count = 60000; 
    int epochs = 10; 
    int batch_size = 128;
    // int num_classes = labels->shape[1];

    // Create the CNN instead of the MLP
    SimpleCNN* model = create_simple_cnn(device);
    if (!model) return NULL;

    int num_params;
    Tensor** params = cnn_get_parameters(model, &num_params);
    
    // Adam is good with lr around 0.001
    Adam* optimizer = adam_create(params, num_params, 0.001f); 

    printf("\n[%s] Starting CNN training on %d samples...\n", label, sample_count);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = sample_count / batch_size;

        for (int b = 0; b < num_batches; b++) {
            int start_row = b * batch_size;
            Tensor* batch_inputs = tensor_slice_view(images, start_row, batch_size);
            Tensor* batch_labels = tensor_slice_view(labels, start_row, batch_size);

            // --- THE RESHAPE TRICK FOR TRAINING ---
            batch_inputs->ndims = 4;
            free(batch_inputs->shape);
            batch_inputs->shape = (int*)malloc(4 * sizeof(int));
            batch_inputs->shape[0] = batch_size;
            batch_inputs->shape[1] = 1;
            batch_inputs->shape[2] = 28;
            batch_inputs->shape[3] = 28;

            // Forward -> Loss -> Backward -> Step
            Tensor* predictions = cnn_forward(model, batch_inputs);
            Tensor* loss = tensor_cross_entropy(predictions, batch_labels);

            total_loss += tensor_scalar_value(loss);

            seed_loss_grad(loss);
            backward(loss);
            adam_step(optimizer);
            adam_zero_grad(optimizer);

            free_graph(loss);
            free_tensor(batch_inputs);
            free_tensor(batch_labels);
        }

        printf("[%s] Epoch %d/%d | Average Loss: %.4f\n", label, epoch + 1, epochs, total_loss / num_batches);
    }

    adam_free(optimizer);
    free(params); 
    
    return model;
}

int main(void) {
    printf("--- SmallGrad CNN MNIST Training & Testing Demo ---\n\n");

    printf("Loading MNIST Training data...\n");
    Tensor* train_images = load_mnist_images("data/train-images-idx3-ubyte");
    Tensor* train_labels = load_mnist_labels("data/train-labels-idx1-ubyte");

    printf("Loading MNIST Testing data...\n");
    Tensor* test_images = load_mnist_images("data/t10k-images-idx3-ubyte");
    Tensor* test_labels = load_mnist_labels("data/t10k-labels-idx1-ubyte");

    if (!train_images || !train_labels || !test_images || !test_labels) {
        fprintf(stderr, "Error: failed to load datasets. Check your data/ folder.\n");
        return 1;
    }
    
    printf("\nTransferring all datasets to GPU VRAM...\n");
    tensor_to_device(train_images, DEVICE_GPU);
    tensor_to_device(train_labels, DEVICE_GPU);
    tensor_to_device(test_images, DEVICE_GPU);
    tensor_to_device(test_labels, DEVICE_GPU);
    
    // 1. Train the model
    SimpleCNN* trained_model = run_simple_training(DEVICE_GPU, "GPU", train_images, train_labels);

    // 2. Test the model
    if (trained_model) {
        evaluate_model(trained_model, test_images, test_labels);
        free_simple_cnn(trained_model); 
    }

    // Clean up global data
    free_tensor(train_images);
    free_tensor(train_labels);
    free_tensor(test_images);
    free_tensor(test_labels);

    printf("Demo completed successfully!\n");
    return 0;
}
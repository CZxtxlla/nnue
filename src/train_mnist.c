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
void evaluate_model(MLP* model, Tensor* test_images, Tensor* test_labels) {
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
        
        // FORWARD PASS ONLY (No backward, no optimizer step!)
        Tensor* predictions = mlp_forward(model, batch_inputs);
        Tensor* loss = tensor_cross_entropy(predictions, batch_labels);
        
        total_loss += tensor_scalar_value(loss);
        
        // Download predictions back to CPU to calculate accuracy
        float* host_preds = (float*)malloc(predictions->size * sizeof(float));
        float* host_labels = (float*)malloc(batch_labels->size * sizeof(float));
        
        tensor_download_data(predictions, host_preds);
        tensor_download_data(batch_labels, host_labels);
        
        // Compare predictions to targets
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
    }
    
    float accuracy = ((float)correct_predictions / total_evaluated) * 100.0f;
    printf("\n--- Test Results ---\n");
    printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct_predictions, total_evaluated);
    printf("Average Loss: %.4f\n--------------------\n\n", total_loss / num_batches);
}

// --- The Training Loop (Modified to return the trained model) ---
MLP* run_simple_training(DeviceType device, const char* label, Tensor* images, Tensor* labels) {
    int sample_count = 60000; 
    int epochs = 10; 
    int batch_size = 128;
    int input_features = images->shape[1];
    int num_classes = labels->shape[1];

    int architecture[] = {input_features, 128, 64, num_classes};
    MLP* model = create_mlp(architecture, 4, device);
    if (!model) return NULL;

    int num_params;
    Tensor** params = mlp_get_parameters(model, &num_params);
    SGD* optimizer = sgd_create(params, num_params, 0.05f);

    printf("\n[%s] Starting training on %d samples...\n", label, sample_count);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = sample_count / batch_size;

        for (int b = 0; b < num_batches; b++) {
            int start_row = b * batch_size;
            Tensor* batch_inputs = tensor_slice_view(images, start_row, batch_size);
            Tensor* batch_labels = tensor_slice_view(labels, start_row, batch_size);

            Tensor* predictions = mlp_forward(model, batch_inputs);
            Tensor* loss = tensor_cross_entropy(predictions, batch_labels);

            total_loss += tensor_scalar_value(loss);

            seed_loss_grad(loss);
            backward(loss);
            sgd_step(optimizer);
            sgd_zero_grad(optimizer);

            free_graph(loss);
        }

        printf("[%s] Epoch %d/%d | Average Loss: %.4f\n", label, epoch + 1, epochs, total_loss / num_batches);
    }

    sgd_free(optimizer);
    free(params); // Frees the array of pointers, not the tensors themselves
    
    return model; // Return the trained model
}

int main(void) {
    printf("--- SmallGrad MNIST Training & Testing Demo ---\n\n");

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

    // Train the model
    MLP* trained_model = run_simple_training(DEVICE_GPU, "GPU", train_images, train_labels);

    // Test the model
    if (trained_model) {
        printf("\n--- Evaluating Original Trained Model ---\n");
        evaluate_model(trained_model, test_images, test_labels);

        char* filepath = "mnist_model.smlp";
        printf("Saving model to %s\n", filepath);
        if (save_mlp(trained_model, filepath)) {
            printf("Model saved successfully!\n");
        } else {
            fprintf(stderr, "failed to save model.\n");
        }

        free_mlp(trained_model); // Free the model now that we are done testing

        MLP* loaded_model = load_mlp(filepath, DEVICE_GPU);

        if (loaded_model) {
            printf("Model loaded successfully!\n");
            
            // Evaluate the loaded model
            printf("\n--- Evaluating Loaded Model ---\n");
            evaluate_model(loaded_model, test_images, test_labels);
            
            // clean up the loaded model
            free_mlp(loaded_model);
        } else {
            fprintf(stderr, "Failed to load model from %s\n", filepath);
        }
    }

    // Clean up global data
    free_tensor(train_images);
    free_tensor(train_labels);
    free_tensor(test_images);
    free_tensor(test_labels);

    printf("Demo completed successfully!\n");
    return 0;
}
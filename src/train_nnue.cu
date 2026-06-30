#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>

extern "C" void init_framework();
extern "C" void cleanup_framework();

#define MAX_ACTIVE 32
#define IN_FEATURES 41024
#define ACCUMULATOR_SIZE 256

typedef struct {
    void* mmap_ptr;
    size_t filesize;
    int num_samples;
    size_t sample_size;
} NnueDataset;

NnueDataset* load_nnue_dataset(const char* filepath) {
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) return NULL;

    struct stat st;
    fstat(fd, &st);
    size_t filesize = st.st_size;

    void* mmap_ptr = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_ptr == MAP_FAILED) return NULL;

    size_t sample_size = (MAX_ACTIVE * sizeof(int)) * 2 + sizeof(int) + sizeof(float);
    
    NnueDataset* data = (NnueDataset*)malloc(sizeof(NnueDataset));
    data->num_samples = filesize / sample_size;
    data->mmap_ptr = mmap_ptr;
    data->filesize = filesize;
    data->sample_size = sample_size;

    return data;
}

void free_nnue_dataset(NnueDataset* data) {
    if (data) {
        munmap(data->mmap_ptr, data->filesize);
        free(data);
    }
}

NNUE* run_nnue_training(DeviceType device, const char* label, NnueDataset* dataset) {
    int epochs = 1; 
    int batch_size = 4096; 
    int num_batches = dataset->num_samples / batch_size;

    int hidden_dims[] = {ACCUMULATOR_SIZE * 2, 32, 32, 1}; 
    NNUE* model = create_nnue(IN_FEATURES, ACCUMULATOR_SIZE, hidden_dims, 3);
    
    int num_params;
    Tensor** params = nnue_get_parameters(model, &num_params);
    Adam* optimizer = adam_create(params, num_params, 0.001f);

    int batch_shape_active[] = {batch_size, MAX_ACTIVE};
    int batch_shape_scalar[] = {batch_size, 1};

    Tensor* batch_w = create_tensor(batch_shape_active, 2, DEVICE_GPU, false, 1);
    Tensor* batch_b = create_tensor(batch_shape_active, 2, DEVICE_GPU, false, 1);
    Tensor* batch_stm = create_tensor(batch_shape_scalar, 2, DEVICE_GPU, false, 1);
    Tensor* batch_score = create_tensor(batch_shape_scalar, 2, DEVICE_GPU, false, 0);

    // Temp host buffer for de-interleaving the batch
    int* h_w = (int*)malloc(batch_size * MAX_ACTIVE * sizeof(int));
    int* h_b = (int*)malloc(batch_size * MAX_ACTIVE * sizeof(int));
    int* h_stm = (int*)malloc(batch_size * sizeof(int));
    float* h_score = (float*)malloc(batch_size * sizeof(float));

    printf("\n[%s] Starting training on %d samples...\n", label, dataset->num_samples);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (int b = 0; b < num_batches; b++) {
            // De-interleave the binary data into the host buffers
            for (int i = 0; i < batch_size; i++) {
                char* ptr = (char*)dataset->mmap_ptr + ((b * batch_size + i) * dataset->sample_size);
                memcpy(&h_w[i * MAX_ACTIVE], ptr, MAX_ACTIVE * sizeof(int)); ptr += MAX_ACTIVE * sizeof(int);
                memcpy(&h_b[i * MAX_ACTIVE], ptr, MAX_ACTIVE * sizeof(int)); ptr += MAX_ACTIVE * sizeof(int);
                memcpy(&h_stm[i], ptr, sizeof(int)); ptr += sizeof(int);
                memcpy(&h_score[i], ptr, sizeof(float));
            }

            cudaMemcpy(batch_w->device_int_data, h_w, batch_size * MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(batch_b->device_int_data, h_b, batch_size * MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(batch_stm->device_int_data, h_stm, batch_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(batch_score->gpu_data, h_score, batch_size * sizeof(float), cudaMemcpyHostToDevice);

            Tensor* predictions = nnue_forward(model, batch_w, batch_b, batch_stm);
            Tensor* loss = tensor_blended_loss_forward(predictions, batch_score, batch_score, 1.0f);

            total_loss += tensor_scalar_value(loss);
            seed_loss_grad(loss);
            backward(loss);
            adam_step(optimizer);
            adam_zero_grad(optimizer);
            free_graph(loss);
        }
        printf("[%s] Epoch %d/%d | Avg Loss: %.6f\n", label, epoch + 1, epochs, total_loss / num_batches);
    }

    free(h_w); free(h_b); free(h_stm); free(h_score);
    adam_free(optimizer); free(params);
    free_tensor(batch_w); free_tensor(batch_b); free_tensor(batch_stm); free_tensor(batch_score);
    return model;
}

int main(void) {
    init_framework();
    NnueDataset* dataset = load_nnue_dataset("data_handling/bootstrapping_data.bin");
    if (!dataset) return 1;
    NNUE* trained_model = run_nnue_training(DEVICE_GPU, "GPU", dataset);
    if (trained_model) {
        save_nnue(trained_model, "halfkp_model.nnue");
        free_nnue(trained_model);
    }
    free_nnue_dataset(dataset);
    cleanup_framework();
    return 0;
}
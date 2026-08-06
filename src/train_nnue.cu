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
#include <stdint.h>
#include <math.h>
#include <time.h>

extern "C" void init_framework();
extern "C" void cleanup_framework();

#define MAX_ACTIVE 32
#define IN_FEATURES 768
#define ACCUMULATOR_SIZE 1024

static int flip_sq(int sq) {
    return sq ^ 56;
}

static int flip_piece(int p_type) {
    return (p_type + 6) % 12;
}

typedef struct {
    void* mmap_ptr;
    size_t filesize;
    int num_samples;
    size_t sample_size;
} NnueDataset;

void free_nnue_dataset(NnueDataset* data) {
    if (data) {
        munmap(data->mmap_ptr, data->filesize);
        free(data);
    }
}

// (74 bytes)
#pragma pack(push, 1)
typedef struct {
    int16_t eval; 
    uint16_t win;
    uint16_t draw;
    uint16_t loss;
    uint8_t stm; 
    uint8_t num_features;
    uint16_t features[32]; 
} Sample;
#pragma pack(pop)

// Dataset loader
NnueDataset* load_nnue_dataset(const char* filepath) {
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) return NULL;

    struct stat st;
    fstat(fd, &st);
    size_t filesize = st.st_size;

    void* mmap_ptr = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_ptr == MAP_FAILED) return NULL;

    // Use sizeof(Sample) to guarantee exact 74-byte strides
    NnueDataset* data = (NnueDataset*)malloc(sizeof(NnueDataset));
    data->sample_size = sizeof(Sample); 
    data->num_samples = filesize / data->sample_size;
    data->mmap_ptr = mmap_ptr;
    data->filesize = filesize;

    return data;
}

NNUE* run_nnue_training(DeviceType device, const char* label, const char** filepaths, const char* val_filepath, int num_files, float lambda, float lr, float K, NNUE* existing_model) {
    int epochs = 20; 
    int batch_size = 16384; 

    int drop_every_n_epochs = 5; 
    float drop_factor = 0.5f; 

    NNUE* model;
    int hidden_dims[] = {ACCUMULATOR_SIZE * 2, 1}; 
    if (existing_model == NULL) {
        // create model for training if one wasn't passed
        model = create_nnue(IN_FEATURES, ACCUMULATOR_SIZE, hidden_dims, 1);
    } else {
        model = existing_model;
    }
    
    int num_params;
    Tensor** params = nnue_get_parameters(model, &num_params);
    Adam* optimizer = adam_create(params, num_params, lr);

    int batch_shape_active[] = {batch_size, MAX_ACTIVE};
    int batch_shape_scalar[] = {batch_size, 1};

    // Allocate GPU Tensors and Host Buffers
    Tensor* batch_w = create_tensor(batch_shape_active, 2, DEVICE_GPU, false, 1);
    Tensor* batch_b = create_tensor(batch_shape_active, 2, DEVICE_GPU, false, 1);
    Tensor* batch_stm = create_tensor(batch_shape_scalar, 2, DEVICE_GPU, false, 1);
    Tensor* batch_score = create_tensor(batch_shape_scalar, 2, DEVICE_GPU, false, 0);

    int* h_w = (int*)malloc(batch_size * MAX_ACTIVE * sizeof(int));
    int* h_b = (int*)malloc(batch_size * MAX_ACTIVE * sizeof(int));
    int* h_stm = (int*)malloc(batch_size * sizeof(int));
    float* h_score = (float*)malloc(batch_size * sizeof(float));

    printf("\n[%s] Starting training across %d dataset files...\n", label, num_files);

    NnueDataset* val_dataset = load_nnue_dataset(val_filepath);
    if (!val_dataset) {
        printf("Warning: Could not load validation set %s. Validation will be skipped.\n", val_filepath);
    }

    // Array to hold the shuffled file indices
    int* file_indices = (int*)malloc(num_files * sizeof(int));
    for(int i = 0; i < num_files; i++) {
        file_indices[i] = i;
    }

    // Seed the random generator for shuffling
    srand((unsigned int)time(NULL));

    for (int epoch = 0; epoch < epochs; epoch++) {
        
        if (epoch > 0 && epoch % drop_every_n_epochs == 0) {
            optimizer->lr *= drop_factor;
            printf("\n>>> [Scheduler] Learning Rate reduced to: %f <<<\n\n", optimizer->lr);
        }
        

        // Shuffle the order of the dataset chunks
        for (int i = num_files - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = file_indices[i];
            file_indices[i] = file_indices[j];
            file_indices[j] = temp;
        }
        
        float epoch_total_loss = 0.0f;
        int epoch_total_batches = 0;
        float val_total_loss = 0.0f;
        int val_total_batches = 0;

        // loop through dataset files
        for (int f = 0; f < num_files; f++) {
            int file_idx = file_indices[f];
            NnueDataset* dataset = load_nnue_dataset(filepaths[file_idx]);
            
            if (!dataset) {
                printf("Warning: Could not load %s. Skipping...\n", filepaths[file_idx]);
                continue;
            }

            int num_batches = dataset->num_samples / batch_size;
            
            for (int b = 0; b < num_batches; b++) {
                
                Sample* batch_samples = (Sample*)((char*)dataset->mmap_ptr + (b * batch_size * sizeof(Sample)));
                
                for (int i = 0; i < batch_size; i++) {
                    Sample* s = &batch_samples[i];
                    
                    float wdl_target = (float)s->win + (float)s->draw / 2.0f; //(s->win + (s->draw / 2.0f)) / 1000.0f; uncomment if stockfish data
                    float eval_target = 1.0f / (1.0f + expf(-(float)s->eval / K));
                    float absolute_score = (lambda * eval_target) + ((1.0f - lambda) * wdl_target);
                    
                    h_stm[i] = s->stm;

                    /*
                    uncomment if training on stockfish data since the eval is absolute, 
                    self play data the eval is already based on stm
                    if (s->stm == 1) { 
                        // If it is black's turn, flip the probability
                        h_score[i] = 1.0f - absolute_score; 
                    } else {
                        h_score[i] = absolute_score;
                    }
                    */

                    h_score[i] = absolute_score;
                    
                    for(int feat = 0; feat < MAX_ACTIVE; feat++) {
                        int feature_val = (s->features[feat] == 65535) ? -1 : (int)s->features[feat];
                        
                        h_w[i * MAX_ACTIVE + feat] = feature_val;

                        if (feature_val < 0) {
                            h_b[i * MAX_ACTIVE + feat] = -1;
                        } else {
                            int piece_idx = feature_val / 64;
                            int square_idx = feature_val % 64;
                            h_b[i * MAX_ACTIVE + feat] = (flip_piece(piece_idx) * 64) + flip_sq(square_idx);
                        }
                    }
                }

                cudaMemcpy(batch_w->device_int_data, h_w, batch_size * MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(batch_b->device_int_data, h_b, batch_size * MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(batch_stm->device_int_data, h_stm, batch_size * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(batch_score->gpu_data, h_score, batch_size * sizeof(float), cudaMemcpyHostToDevice);

                Tensor* raw_logits = nnue_forward(model, batch_w, batch_b, batch_stm);
                Tensor* predictions = tensor_sigmoid(raw_logits);
                Tensor* loss = tensor_mse(predictions, batch_score);

                epoch_total_loss += tensor_scalar_value(loss);
                seed_loss_grad(loss);
                backward(loss);
                adam_step(optimizer);
                adam_zero_grad(optimizer);
                free_graph(loss);
            }
            
            // Add to the total batch count and close the file chunk
            epoch_total_batches += num_batches;
            free_nnue_dataset(dataset);
        }

        if (val_dataset) {
            int num_val_batches = val_dataset->num_samples / batch_size;

            for (int b = 0; b < num_val_batches; b++) {
                Sample* batch_samples = (Sample*)((char*)val_dataset->mmap_ptr + (b * batch_size * sizeof(Sample)));

                for (int i = 0; i < batch_size; i++) {
                    Sample* s = &batch_samples[i];

                    float wdl_target =  (float)s->win + (float)s->draw / 2.0f; // (s->win + (s->draw / 2.0f)) / 1000.0f;
                    float eval_target = 1.0f / (1.0f + expf(-(float)s->eval / K));
                    float absolute_score = (lambda * eval_target) + ((1.0f - lambda) * wdl_target);
                    
                    h_stm[i] = s->stm;

                    /*
                    if (s->stm == 1) {
                        h_score[i] = 1.0f - absolute_score;
                    } else {
                        h_score[i] = absolute_score;
                    }
                    */

                    h_score[i] = absolute_score;

                    for (int feat = 0; feat < MAX_ACTIVE; feat++) {
                        int feature_val = (s->features[feat] == 65535) ? -1 : (int)s->features[feat];
                        h_w[i * MAX_ACTIVE + feat] = feature_val;

                        if (feature_val < 0) {
                            h_b[i * MAX_ACTIVE + feat] = -1;
                        } else {
                            int piece_idx = feature_val / 64;
                            int square_idx = feature_val % 64;
                            h_b[i * MAX_ACTIVE + feat] = (flip_piece(piece_idx) * 64) + flip_sq(square_idx);
                        }
                    }
                }

                cudaMemcpy(batch_w->device_int_data, h_w, batch_size * MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(batch_b->device_int_data, h_b, batch_size * MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(batch_stm->device_int_data, h_stm, batch_size * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(batch_score->gpu_data, h_score, batch_size * sizeof(float), cudaMemcpyHostToDevice);

                Tensor* raw_logits = nnue_forward(model, batch_w, batch_b, batch_stm);
                Tensor* predictions = tensor_sigmoid(raw_logits);
                Tensor* loss = tensor_mse(predictions, batch_score);

                val_total_loss += tensor_scalar_value(loss);
                free_graph(loss);
            }

            val_total_batches = num_val_batches;
        }
        
        if (epoch_total_batches > 0 && val_total_batches > 0) {
            float train_loss_avg = epoch_total_loss / epoch_total_batches;
            float val_loss_avg = val_total_loss / val_total_batches;

            printf("[%s] Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f\n",
                   label, epoch + 1, epochs, train_loss_avg, val_loss_avg);
        } else if (epoch_total_batches > 0) {
            printf("[%s] Epoch %d/%d | Train Loss: %.6f\n", label, epoch + 1, epochs, epoch_total_loss / epoch_total_batches);
        } else {
            printf("[%s] Epoch %d/%d | No data processed.\n", label, epoch + 1, epochs);
        }
    }

    if (val_dataset) {
        free_nnue_dataset(val_dataset);
    }

    free(h_w); free(h_b); free(h_stm); free(h_score);
    adam_free(optimizer); free(params);
    free_tensor(batch_w); free_tensor(batch_b); free_tensor(batch_stm); free_tensor(batch_score);
    free(file_indices);
    
    return model;
}


int main(int argc, char* argv[]) {

    //safety check
    if (argc < 4) {
        printf("Usage: %s <lambda> <lr> <K>\n", argv[0]);
        return 1;
    }

    float lambda_val = atof(argv[1]);
    float lr_val = atof(argv[2]);
    float k_val = atof(argv[3]);

    init_framework();
    
    // dataset files
    /*
    const char* datasets[] = {
        "data_handling/training_data_part_0.bin",
        "data_handling/training_data_part_1.bin",
        "data_handling/training_data_part_2.bin",
        "data_handling/training_data_part_3.bin",
        "data_handling/training_data_part_4.bin",
        "data_handling/training_data_part_5.bin",
        "data_handling/training_data_part_6.bin",
        "data_handling/training_data_part_7.bin",
        "data_handling/training_data_part_8.bin",
        "data_handling/training_data_part_9.bin"
        //"data_handling/training_data_part_10.bin"
    };
    */
    const char* datasets[] = {
        "selfplay_data/training_data_thread_0.bin",
        "selfplay_data/training_data_thread_1.bin",
        "selfplay_data/training_data_thread_2.bin",
        "selfplay_data/training_data_thread_3.bin",
        "selfplay_data/training_data_thread_4.bin",
        "selfplay_data/training_data_thread_5.bin",
        "selfplay_data/training_data_thread_6.bin",
        "selfplay_data/training_data_thread_7.bin",
        "selfplay_data/training_data_thread_8.bin",
        //"selfplay_data/training_data_thread_9.bin",
    };

    
    int num_datasets = sizeof(datasets) / sizeof(datasets[0]);

    const char* validation_dataset = "selfplay_data/training_data_thread_9.bin";

    NNUE* existing_model = load_nnue("768_float_9_18_50_1024.nnue", DEVICE_GPU); // already trained model

    NNUE* trained_model = run_nnue_training(DEVICE_GPU, "GPU", datasets, validation_dataset /*validation dataset*/, num_datasets, lambda_val, lr_val, k_val, NULL);
    
    if (trained_model) {
        save_nnue(trained_model, "768_float_50_1024_v1.nnue"); // v1 = trained on Mark_11 (hc positional eval)
        save_nnue_quantized(trained_model, "768_quant_50_1024_v1.nnue");
        free_nnue(trained_model);
    }
    
    cleanup_framework();
    return 0;
}
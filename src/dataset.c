#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/dataset.h"

static uint32_t reverse_int(uint32_t i) {
    // helper to swap Big endian to little endian
    return ((i & 0xff000000) >> 24) | ((i & 0x00ff0000) >> 8) | ((i & 0x0000ff00) << 8 | ((i & 0x000000ff) << 24));
}

Tensor* load_mnist_images(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open MNIST images file.\n");
        return NULL;
    }

    uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    // read the header
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_images, sizeof(uint32_t), 1, file);
    fread(&num_rows, sizeof(uint32_t), 1, file);
    fread(&num_cols, sizeof(uint32_t), 1, file);

    // fix endian
    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);

    if (magic_number != 2051) {
        fprintf(stderr, "Error: invalid MNIST magic number.\n");
        fclose(file);
        return NULL;
    }

    int pixels_per_image = num_rows * num_cols;
    int shape[] = {num_images, pixels_per_image};

    Tensor* images = create_tensor(shape, 2, DEVICE_CPU, false);

    // read raw pixels
    unsigned char* raw_pixels = (unsigned char*)malloc(num_images * pixels_per_image);
    fread(raw_pixels, sizeof(unsigned char), num_images * pixels_per_image, file);

    // normalize from 0 to 1 and load into tensor data
    for (int i = 0; i < images->size; i++) {
        images->cpu_data[i] = (float)raw_pixels[i] / 255.0f;
    }

    free(raw_pixels);
    fclose(file);
    return images;
}

Tensor* load_mnist_labels(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open MNIST labels file.\n");
        return NULL;
    }

    uint32_t magic_number = 0, num_items = 0;

    // Read the header
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_items, sizeof(uint32_t), 1, file);

    // fix endian
    magic_number = reverse_int(magic_number);
    num_items = reverse_int(num_items);

    if (magic_number != 2049) {
        fprintf(stderr, "Error: Invalid MNIST labels magic number.\n");
        fclose(file);
        return NULL;
    }

    int num_classes = 10;
    int shape[] = {num_items, num_classes};

    Tensor* labels = create_tensor(shape, 2, DEVICE_CPU, false);

    unsigned char* raw_labels = (unsigned char*)malloc(num_items);
    fread(raw_labels, sizeof(unsigned char), num_items, file);

    // convert labels into one-hot-encoding
    for (int i = 0; i < num_items; i++) {
        int label_val = raw_labels[i];
        for (int j = 0; j < num_classes; j++) {
            labels->cpu_data[i * num_classes + j] = (j == label_val) ? 1.0f : 0.0f;
        }
    }

    free(raw_labels);
    fclose(file);
    return labels;
}
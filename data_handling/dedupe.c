#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

#define HASH_TABLE_SIZE 268435456ULL // 2^28 slots, 2.14 GB ram

uint64_t* hash_table;

uint64_t hash_features(uint16_t* features) {
    uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < 32; i++) {
        hash ^= features[i];
        hash *= 1099511628211ULL;
    }
    if (hash == 0) hash = 1; // Reserve 0 to mean "empty slot"
    return hash;
}

int is_unique_and_insert(uint64_t hash) {
    uint64_t idx = hash & (HASH_TABLE_SIZE - 1); 
    
    while (hash_table[idx] != 0) {
        if (hash_table[idx] == hash) {
            return 0; // seen before
        }
        idx = (idx + 1) & (HASH_TABLE_SIZE - 1);
    }
    
    hash_table[idx] = hash;
    return 1; // unique
}

void process_file(const char* filepath, FILE* out_file, long long* total, long long* unique) {
    FILE* in_file = fopen(filepath, "rb");
    if (!in_file) {
        printf("Could not open %s, skipping...\n", filepath);
        return;
    }

    Sample s;
    while (fread(&s, sizeof(Sample), 1, in_file) == 1) {
        (*total)++;
        
        uint64_t hash = hash_features(s.features);
        
        if (is_unique_and_insert(hash)) {
            fwrite(&s, sizeof(Sample), 1, out_file);
            (*unique)++;
        }
    }
    fclose(in_file);
    printf("Finished %s\n", filepath);
}

int main() {
    printf("Allocating 2GB Hash Table...\n");
    hash_table = (uint64_t*)calloc(HASH_TABLE_SIZE, sizeof(uint64_t));
    if (!hash_table) {
        printf("Failed to allocate memory! Close some Chrome tabs.\n");
        return 1;
    }

    FILE* out_file = fopen("combined_unique_dataset.bin", "wb");
    if (!out_file) return 1;

    long long total_positions = 0;
    long long unique_positions = 0;

    // nnue files first for higher priority
    const char* files[] = {
        // --- GEN 1 (NNUE DATA) ---
        "selfplay_data_v2/training_data_thread_0.bin",
        "selfplay_data_v2/training_data_thread_1.bin",
        "selfplay_data_v2/training_data_thread_2.bin",
        "selfplay_data_v2/training_data_thread_3.bin",
        "selfplay_data_v2/training_data_thread_4.bin",
        "selfplay_data_v2/training_data_thread_5.bin",
        "selfplay_data_v2/training_data_thread_6.bin",
        "selfplay_data_v2/training_data_thread_7.bin",
        "selfplay_data_v2/training_data_thread_8.bin",
        "selfplay_data_v2/training_data_thread_9.bin",
        
        // --- GEN 0 (HCE DATA) ---
        "selfplay_data_v1/training_data_thread_0.bin",
        "selfplay_data_v1/training_data_thread_1.bin",
        "selfplay_data_v1/training_data_thread_2.bin",
        "selfplay_data_v1/training_data_thread_3.bin",
        "selfplay_data_v1/training_data_thread_4.bin",
        "selfplay_data_v1/training_data_thread_5.bin",
        "selfplay_data_v1/training_data_thread_6.bin",
        "selfplay_data_v1/training_data_thread_7.bin",
        "selfplay_data_v1/training_data_thread_8.bin",
        "selfplay_data_v1/training_data_thread_9.bin"
    };
    
    int num_files = sizeof(files) / sizeof(files[0]);

    for (int i = 0; i < num_files; i++) {
        process_file(files[i], out_file, &total_positions, &unique_positions);
    }

    fclose(out_file);
    free(hash_table);

    printf("\n=== DEDUPLICATION COMPLETE ===\n");
    printf("Total Positions Scanned: %lld\n", total_positions);
    printf("Unique Positions Saved : %lld\n", unique_positions);
    printf("Duplicates Destroyed   : %lld\n", total_positions - unique_positions);

    return 0;
}
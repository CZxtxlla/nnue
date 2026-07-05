#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"
#include "../include/nn.h"

extern "C" void init_framework();
extern "C" void cleanup_framework();

#define MAX_ACTIVE 32

// --- FEN Parsing Helpers (Updated for 768) ---

// 0-5 for White, 6-11 for Black (Matches python-chess: P=0, K=5, p=6, k=11)
int char_to_piece(char c) {
    switch (c) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2; case 'R': return 3; case 'Q': return 4; case 'K': return 5;
        case 'p': return 6; case 'n': return 7; case 'b': return 8; case 'r': return 9; case 'q': return 10; case 'k': return 11;
        default: return -1;
    }
}

// Flip square vertically (A1 -> A8)
int flip_sq(int sq) { return sq ^ 56; }

// Swap White (0-5) and Black (6-11)
int flip_piece(int p_type) { return (p_type + 6) % 12; }

// --- Evaluation Function ---
void evaluate_fen(NNUE* model, const char* fen) {
    int w_idx[MAX_ACTIVE], b_idx[MAX_ACTIVE];
    int stm;
    int pieces[32], squares[32];
    int num_pieces = 0;

    // Initialize arrays to -1 (empty)
    for (int i = 0; i < MAX_ACTIVE; i++) {
        w_idx[i] = -1;
        b_idx[i] = -1;
    }

    // 1. Parse the FEN string
    char fen_copy[256];
    strncpy(fen_copy, fen, 256);
    
    char* token = strtok(fen_copy, " ");
    int sq = 56; // Start at a8 (Index 56)
    
    for (int i = 0; token[i] != '\0'; i++) {
        char c = token[i];
        if (c == '/') sq -= 16;
        else if (isdigit(c)) sq += (c - '0');
        else {
            int p_type = char_to_piece(c);
            if (p_type != -1) {
                pieces[num_pieces] = p_type;
                squares[num_pieces] = sq++;
                num_pieces++;
            }
        }
    }

    token = strtok(NULL, " ");
    stm = (token && token[0] == 'b') ? 1 : 0;

    // 2. Map to 768 Indices
    for (int i = 0; i < num_pieces && i < MAX_ACTIVE; i++) {
        int p = pieces[i];
        int s = squares[i];
        
        // White's Perspective
        w_idx[i] = (p * 64) + s;
        
        // Black's Perspective (Flip the board visually and swap piece colors)
        b_idx[i] = (flip_piece(p) * 64) + flip_sq(s);
    }

    // 3. Setup GPU Tensors (Batch size = 1)
    int active_shape[] = {1, MAX_ACTIVE};
    int scalar_shape[] = {1, 1};

    Tensor* t_w = create_tensor(active_shape, 2, DEVICE_GPU, false, 1);
    Tensor* t_b = create_tensor(active_shape, 2, DEVICE_GPU, false, 1);
    Tensor* t_stm = create_tensor(scalar_shape, 2, DEVICE_GPU, false, 1);

    // Upload data to VRAM
    cudaMemcpy(t_w->device_int_data, w_idx, MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(t_b->device_int_data, b_idx, MAX_ACTIVE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(t_stm->device_int_data, &stm, sizeof(int), cudaMemcpyHostToDevice);

    Tensor* out = nnue_forward(model, t_w, t_b, t_stm);

    float raw_logit;
    tensor_download_data(out, &raw_logit);

    float stm_win_prob = raw_logit;
    if (stm_win_prob < 0.001f) stm_win_prob = 0.001f;
    if (stm_win_prob > 0.999f) stm_win_prob = 0.999f;

    float centipawns = -400.0f * logf((1.0f / stm_win_prob) - 1.0f);
    if (stm == 1) {
        centipawns = -centipawns; 
    }

    // 5. Output
    printf("\nFEN: %s\n", fen);
    printf("Side to move: %s\n", stm == 1 ? "Black" : "White");
    printf("STM Win Prob:     %.2f%%\n", stm_win_prob * 100.0f);
    printf("Centipawn Eval:   %.2f (%.2f Pawns)\n", centipawns, centipawns / 100.0f);

    // 6. Cleanup Graph and Tensors
    free_graph(out);
    free_tensor(t_w);
    free_tensor(t_b);
    free_tensor(t_stm);
}

int main(void) {
    init_framework();
    
    printf("Loading model...\n");
    // Make sure to load the float checkpoint, not the quantized inference one!
    NNUE* model = load_nnue("768_model_float_9_18.nnue", DEVICE_GPU); 
    if (!model) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    const char* start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const char* white_winning = "4k3/Q7/4K3/8/8/8/8/8 w - - 0 1";
    const char* black_winning = "4k3/8/8/8/8/4k3/q7/4K3 b - - 0 1";

    const char* test_1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    const char* test_2 = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
    const char* test_3 = "r1b2rk1/pp1nb1pp/1qn1p3/3pP3/3P4/P2B1N2/1P1BN1PP/R2QK2R w KQ - 1 14";
    const char* test_4 = "2r4k/p4b2/4pq2/1p1p1nR1/5P2/P2B4/1P2Q2P/1K4R1 b - - 2 30";

    evaluate_fen(model, start_fen);
    evaluate_fen(model, white_winning);
    evaluate_fen(model, black_winning);
    evaluate_fen(model, test_1);
    evaluate_fen(model, test_2);
    evaluate_fen(model, test_3);
    evaluate_fen(model, test_4);

    free_nnue(model);
    cleanup_framework();
    
    return 0;
}
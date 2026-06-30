#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"
#include "../include/nn.h"

// Forward declarations for your framework
extern "C" void init_framework();
extern "C" void cleanup_framework();

#define MAX_ACTIVE 32

// --- FEN Parsing Helpers ---
int char_to_piece(char c) {
    switch (c) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2; case 'R': return 3; case 'Q': return 4;
        case 'p': return 5; case 'n': return 6; case 'b': return 7; case 'r': return 8; case 'q': return 9;
        default: return -1;
    }
}

int flip_sq(int sq) { return sq ^ 56; }
int flip_piece(int p_type) { return (p_type + 5) % 10; }

// --- Evaluation Function ---
void evaluate_fen(NNUE* model, const char* fen) {
    int w_idx[MAX_ACTIVE], b_idx[MAX_ACTIVE];
    int stm;
    int wk_sq = -1, bk_sq = -1;
    int pieces[32], squares[32];
    int num_pieces = 0;

    for (int i = 0; i < MAX_ACTIVE; i++) {
        w_idx[i] = -1;
        b_idx[i] = -1;
    }

    // 1. Parse the FEN string
    char fen_copy[256];
    strncpy(fen_copy, fen, 256);
    
    char* token = strtok(fen_copy, " ");
    int sq = 56;
    for (int i = 0; token[i] != '\0'; i++) {
        char c = token[i];
        if (c == '/') sq -= 16;
        else if (isdigit(c)) sq += (c - '0');
        else if (c == 'K') wk_sq = sq++;
        else if (c == 'k') bk_sq = sq++;
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

    // 2. Map to HalfKP Indices
    for (int i = 0; i < num_pieces && i < MAX_ACTIVE; i++) {
        int p = pieces[i];
        int s = squares[i];
        
        w_idx[i] = (wk_sq * 641) + (p * 64) + s;
        b_idx[i] = (flip_sq(bk_sq) * 641) + (flip_piece(p) * 64) + flip_sq(s);
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

    // Standard Sigmoid for Probability
    float win_prob = 1.0f / (1.0f + expf(-raw_logit));

    // Multiply by 410 to convert the logit into Centipawns
    float centipawns = raw_logit * 410.0f;

    printf("\nFEN: %s\n", fen);
    printf("Side to move: %s\n", stm == 1 ? "Black" : "White");
    printf("Network Logit:     %.4f\n", raw_logit);
    printf("Centipawn Eval:    %.2f (%.2f Pawns)\n", centipawns, centipawns / 100.0f);
    printf("Win Probability:   %.4f\n", win_prob);

    // 6. Cleanup Graph and Tensors
    free_graph(out);
    free_tensor(t_w);
    free_tensor(t_b);
    free_tensor(t_stm);
}

int main(void) {
    init_framework();
    
    printf("Loading model...\n");
    NNUE* model = load_nnue("dead.nnue", DEVICE_GPU);
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
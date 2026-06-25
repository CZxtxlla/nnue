#include "../include/autograd.h"


static int current_pass_id = 0; // used to tell if a node was visited


// --------- Functions to transmit gradient backwards to parents ------------

void backward_add(Tensor* t) {
    // Takes tensor t which is the result of an addition operation and computes gradients for its parents

    if (t->op != OP_ADD || t->num_parents != 2) {
        fprintf(stderr, "Error: backward_add called on tensor that is not the result of an addition operation.\n");
        return;
    }

    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_add(t, a, b);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_add(t, a, b);
    }
}

void backward_mul(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation and computes gradients for its parents

    if (t->op != OP_MUL || t->num_parents != 2) {
        fprintf(stderr, "Error: backward_add called on tensor that is not the result of a multiplication operation.\n");
        return;
    }

    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_mul(t, a, b);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_mul(t, a, b);
    }
}

void backward_add_bias(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation and computes gradients for its parents

    if (t->op != OP_ADDBIAS || t->num_parents != 2) {
        fprintf(stderr, "Error: backward_add called on tensor that is not the result of an add bias operation.\n");
        return;
    }

    Tensor* a = t->parents[0];
    Tensor* bias = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_add_bias(t, a, bias);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_add_bias(t, a, bias);
    }
}

void backward_matmul(Tensor* t) {
    // Takes tensor t which is the result of a matrix multiplication operation and computes gradients for its parents

    if (t->op != OP_MATMUL || t->num_parents != 2) {
        fprintf(stderr, "Error: backward_add called on tensor that is not the result of a matrix multiplication operation.\n");
        return;
    }

    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_matmul(t, a, b);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_matmul(t, a, b);
    }
}

void backward_relu(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation and computes gradients for its parents

    if (t->op != OP_RELU || t->num_parents != 1) {
        fprintf(stderr, "Error: backward_add called on tensor that is not the result of a relu operation.\n");
        return;
    }

    Tensor* a = t->parents[0];

    if (t->device == DEVICE_CPU) {
        backward_cpu_relu(t, a);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_relu(t, a);
    }
}

void backward_flatten(Tensor* t) {
    if (t->op != OP_FLATTEN || t->num_parents != 1) {
        fprintf(stderr, "Error: backward_flatten called on tensor that is not the result of a flatten operation.\n");
        return;
    }

    Tensor* a = t->parents[0];

    if (t->device == DEVICE_CPU) {
        backward_cpu_flatten(t, a);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_flatten(t, a);
    }
}

void backward_conv2d(Tensor* t) {
    if (t->op != OP_CONV2D || t->num_parents != 3) {
        fprintf(stderr, "Error: backward_conv2d called on tensor that is not the result of a conv operation.\n");
        return;
    }

    Tensor* input = t->parents[0];
    Tensor* weight = t->parents[1];
    Tensor* bias = t->parents[2];

    if (t->device == DEVICE_CPU) {
        backward_cpu_conv2d(t, input, weight, bias);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_conv2d(t, input, weight, bias);
    }
}

void backward_maxpool2d(Tensor* t) {
    if (t->op != OP_MAXPOOL2D || t->num_parents != 1) {
        fprintf(stderr, "Error: backward_maxpool2d called on tensor that is not the result of maxpool operation.\n");
        return;
    }
    
    Tensor* input = t->parents[0];
    
    if (t->device == DEVICE_CPU) {
        backward_cpu_maxpool2d(t, input);
    } else {
        backward_gpu_maxpool2d(t, input);
    }
}

void backward_mse(Tensor* t) {
    // takes tensor t result of mse oper and computes parents gradients
    if (t->op != OP_MSE || t->num_parents != 2) {
        fprintf(stderr, "Error: backward_mse called on tensor that is not the result of a mse operation.\n");
    }

    Tensor* pred = t->parents[0];
    Tensor* target = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_mse(t, pred, target);
    } else {
        backward_gpu_mse(t, pred, target);
    }
}

void backward_cross_entropy(Tensor* t) {
    // takes tensor t result of cross entropy loss and computes parents gradients
    if (t->op != OP_CROSS_ENTROPY || t->num_parents != 2) {
        fprintf(stderr, "Error: backward_mse called on tensor that is not the result of a cross entropy operation.\n");
    }

    Tensor* pred = t->parents[0];
    Tensor* target = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_cross_entropy(t, pred, target);
    } else {
        backward_gpu_cross_entropy(t, pred, target);
    }
}


// ------- Dynamic array implementation -----------

//dynamicly sized array

void tensor_array_init(TensorArray* ta, int initial_capacity) {
    ta->size = 0;
    ta->capacity = initial_capacity;
    ta->array = (Tensor**)malloc(ta->capacity * sizeof(Tensor*));
    if (ta->array == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for tensor array.\n");
        exit(EXIT_FAILURE);
    }
}

void tensor_array_append(TensorArray* ta, Tensor* t) {
    // If we hit the capacity limit, double the capacity
    if (ta->size >= ta->capacity) {
        ta->capacity *= 2;
        ta->array = (Tensor**)realloc(ta->array, ta->capacity * sizeof(Tensor*));

        if (ta->array == NULL) {
            fprintf(stderr, "Error: failed to reallocate memory in tensor_array_append.\n");
            exit(EXIT_FAILURE);
        }
    }
    ta->array[ta->size++] = t;
}

void tensor_array_free(TensorArray* ta) {
    // Free memory allocated for this dynamic array
    free(ta->array);
    ta->size = 0;
    ta->capacity = 0;
}

// ------- Functions for the topological sort -----------

void build_topo(Tensor* u, TensorArray* topo) {
    // Helper, perform DFS and build the topological order of tensors for backpropogation
    if (u->visited_pass_id == current_pass_id) {
        return;
    }

    u->visited_pass_id = current_pass_id;

    for(int i = 0; i < u->num_parents; i++) {
        build_topo(u->parents[i], topo);
    }

    tensor_array_append(topo, u);
}

void collect_nodes(Tensor* root, TensorArray* nodes) {
    if (root == NULL) {
        return;
    }
    if (root->visited_pass_id == current_pass_id) {
        return;
    }

    root->visited_pass_id = current_pass_id; // mark as visited

    for (int i = 0; i < root->num_parents; i++) {
        collect_nodes(root->parents[i], nodes);
    }

    tensor_array_append(nodes, root); // add unvisited node to the collection
}

void free_graph(Tensor* root) {
    if (root == NULL) {
        return;
    }
    current_pass_id++; // make all nodes unvisited

    TensorArray nodes;
    tensor_array_init(&nodes, 16);

    collect_nodes(root, &nodes);

    for (int i = 0; i < nodes.size; i++) {
        Tensor* current = nodes.array[i];

        // Don't free the guys who we provided/weren't created int eh forward pass
        if (current->num_parents > 0) {
            free_tensor(current);
        }
    }

    tensor_array_free(&nodes); // free the collection of tensors
}



// ------ Big boy Backward function -------------


void backward(Tensor* t) {
    current_pass_id++; // mark all nodes as unvisited

    TensorArray topo;
    tensor_array_init(&topo, 16); // initial capacity of 16

    build_topo(t, &topo);

    // seed the final output gradient witrh 1.0, t->grad[0] = 1.0f;

    for (int i = topo.size - 1; i >= 0; i--) {
        Tensor* current = topo.array[i];
        if (current->op == OP_ADD) {
            backward_add(current);
        } else if (current->op == OP_MUL) {
            backward_mul(current);
        } else if (current->op == OP_MATMUL) {
            backward_matmul(current);
        } else if (current->op == OP_ADDBIAS) {
            backward_add_bias(current);
        } else if (current->op == OP_RELU) {
            backward_relu(current);
        } else if (current->op == OP_FLATTEN) {
            backward_flatten(current);
        } else if (current->op == OP_CONV2D) {
            backward_conv2d(current);
        } else if (current->op == OP_MAXPOOL2D) {
            backward_maxpool2d(current);
        } else if (current->op == OP_MSE) {
            backward_mse(current);
        } else if (current->op == OP_CROSS_ENTROPY) {
            backward_cross_entropy(current);
        }

    }

    // free allocated memory
    tensor_array_free(&topo);



}



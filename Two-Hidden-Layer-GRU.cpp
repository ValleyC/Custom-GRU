#include <cmath>
#include <iostream>
#include "WandB.h" // Store weights and biases (can be exported from Pre-trained model in PyTorch)
#include <vector>

const int LOOKBACK = 5; // GRU sequence length (Look-back period)
const int INPUT_SIZE = 7; // Input feature number
const int OUTPUT_SIZE = 5; // Output feature number
const int HIDDEN_SIZE = 40; // Hidden size

typedef float custom_type; // define custom data precision type


// Define activation functions
custom_type sigmoid(custom_type x) {
    return custom_type(1) / (custom_type(1) + std::exp(-x));
}

custom_type relu(custom_type x) {
    return std::max(custom_type(0), x);
}

// first layer of the GRU: input-to-hidden layer
void input_to_hidden(custom_type* input_t, custom_type* hidden) {

    custom_type gi[60] = {0};
    custom_type gh[60] = {0};

    for (int i = 0; i < 60; i++) {
        for (int j = 0; j < 3; j++) {
            gi[i] += input_t[j] * gru_weight_ih_l0[i][j];
        }
        for (int j = 0; j < 20; j++) {
            gh[i] += hidden[j] * gru_weight_hh_l0[i][j];
        }
        gi[i] += gru_bias_ih_l0[i];
        gh[i] += gru_bias_hh_l0[i];
    }

    custom_type resetgate[20] = {0}, inputgate[20] = {0}, newgate[20] = {0};

    for (int i = 0; i < 20; i++) {
        resetgate[i] = sigmoid(gi[i] + gh[i]);
        inputgate[i] = sigmoid(gi[i+20] + gh[i+20]);
        newgate[i] = std::tanh(gi[i+40] + resetgate[i] * gh[i+40]);
        hidden[i] = newgate[i] + inputgate[i] * (hidden[i] - newgate[i]);
    }
}

// second layer of the GRU: hidden-to-hidden layer
void hidden_to_hidden(custom_type* input_t, custom_type* hidden) {

    custom_type gi[HIDDEN_SIZE*3] = {0};
    custom_type gh[HIDDEN_SIZE*3] = {0};

    for (int i = 0; i < HIDDEN_SIZE*3; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            gi[i] += input_t[j] * gru_weight_ih_l1[i][j];
            gh[i] += hidden[j] * gru_weight_hh_l1[i][j];
        }
        gi[i] += gru_bias_ih_l1[i];
        gh[i] += gru_bias_hh_l1[i];
    }

    custom_type resetgate[HIDDEN_SIZE] = {0}, inputgate[HIDDEN_SIZE] = {0}, newgate[HIDDEN_SIZE] = {0};

    for (int i = 0; i < HIDDEN_SIZE; i++) {
            resetgate[i] = sigmoid(gi[i] + gh[i]);
            inputgate[i] = sigmoid(gi[i+HIDDEN_SIZE] + gh[i+HIDDEN_SIZE]);
            newgate[i] = std::tanh(gi[i+HIDDEN_SIZE*2] + resetgate[i] * gh[i+HIDDEN_SIZE*2]);
            hidden[i] = newgate[i] + inputgate[i] * (hidden[i] - newgate[i]);
        }
}

// assemble the GRU network:
// It is assumed that all input data are already after scaling
void gru_network(custom_type input_seq[LOOKBACK][INPUT_SIZE], custom_type (&output)[OUTPUT_SIZE]) {
    custom_type hidden1[HIDDEN_SIZE] = {0};
    custom_type hidden2[HIDDEN_SIZE] = {0};

    // assemble the first two layers:
    for (int t = 0; t < LOOKBACK; t++) {
        input_to_hidden(input_seq[t], hidden1);
        hidden_to_hidden(hidden1, hidden2);
        // repeatedly call the hidden_to_hidden() if there are more hidden layers in the model
        // ex: hidden_to_hidden(hidden2, hidden3);
    }

    // pass the latest hidden state through activation function
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden2[i] = relu(hidden2[i]);
    }

    // fully-connected layer:
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {

            output[i] += fc_weight[i][j]*hidden2[j];
        }
        output[i] += fc_bias[i]; // The output are scaled, remember to unscale it based on the specific scaler used in the GRU training process
    }
}
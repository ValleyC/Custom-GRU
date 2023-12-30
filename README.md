
# GRU Network Implementation in C++

## Overview
This repository contains a C++ implementation of a Gated Recurrent Unit (GRU) neural network. The code is designed to handle time-series data, ideal for applications such as sequence prediction, time-series analysis, and more. The model is structured with two GRU layers followed by a fully connected layer for output.

## Features
- **Custom Data Type Support**: Allows for easy switching between different data types (e.g., float, double, or fixed) for precision adjustments. Useful when designing hardware accelerator in Vitis HLS.
- **Modular GRU Cells**: Separate functions for input-to-hidden and hidden-to-hidden layers.

## Configuration
Before running the model, ensure to configure the following constants as per your requirements:
- `LOOKBACK`: The sequence length for the GRU.
- `INPUT_SIZE`: Number of input features.
- `OUTPUT_SIZE`: Number of output features.
- `HIDDEN_SIZE`: Size of the hidden layers.

## Usage
To use the network:
1. Prepare your input data in the shape `[LOOKBACK][INPUT_SIZE]`.
2. Initialize an array for the output with the shape `[OUTPUT_SIZE]`.
3. Call `gru_network(input_data, output)`.

Example:
```cpp
custom_type input_data[LOOKBACK][INPUT_SIZE] = {/* Your input data */};
custom_type output[OUTPUT_SIZE];

gru_network(input_data, output);
```

## Customization
- You can change the `custom_type` definition to switch between different numerical precision types (e.g., `float`, `double`, or `ap_fixed<32, 8>`).
- Adjust `LOOKBACK`, `INPUT_SIZE`, `OUTPUT_SIZE`, and `HIDDEN_SIZE` to fit the dimensions of your own trained model requirements.

## Weights and Biases
The weights and biases (`WandB.h`) need to be set according to a pre-trained model. This file should contain:
- `gru_weight_ih_l0`, `gru_weight_hh_l0`, `gru_bias_ih_l0`, `gru_bias_hh_l0` for the first GRU layer.
- `gru_weight_ih_l1`, `gru_weight_hh_l1`, `gru_bias_ih_l1`, `gru_bias_hh_l1` for the second GRU layer.
- `fc_weight`, `fc_bias` for the fully connected layer.

## Note
- This code does not include the functionality to train the GRU model. It's designed to work with weights and biases exported from a pre-trained model (e.g., a model trained in PyTorch).
- The input data are assumed to be after-scaling, and therefore the output is also scaled. Ensure the scaling of the input data and output data matches the scale used during the training of the model.

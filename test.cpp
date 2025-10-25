#include <bits/stdc++.h>
using namespace std;

#include "./src/input.cpp"

const int n_input = 28 * 28;
const int n_h1 = 32;
const int n_h2 = 16;
const int n_output = 10;
const int MAX_ITERATIONS = 1;
const double LEARNING_RATE = 0.05;

// Sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the Sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

vector<vector<double>> input_to_h1_weights(n_input, vector<double>(n_h1));
vector<vector<double>> h1_to_h2_weights(n_h1, vector<double>(n_h2));
vector<vector<double>> h2_to_output_weights(n_h2, vector<double>(n_output));

vector<double> biases_h1(n_h1);
vector<double> biases_h2(n_h2);
vector<double> biases_output(n_output);

int main() {
    std::vector<MNISTImage> images = readIDX3Ubyte("train-images.idx3-ubyte");
    std::vector<uint8_t> labels = readIDX1Ubyte("train-labels.idx1-ubyte");
    std::ifstream fin("weights2.txt", std::ios::in);

    for (int i=0; i<n_input; i++) {
        for(int j=0; j<n_h1; j++) {
            fin >> input_to_h1_weights[i][j];
        }
    }

    for (int i=0; i<n_h1; i++) {
        for(int j=0; j<n_h2; j++) {
            fin >> h1_to_h2_weights[i][j];
        }
    }

    for (int i=0; i<n_h2; i++) {
        for(int j=0; j<n_output; j++) {
            fin >> h2_to_output_weights[i][j];
        }
    }
    
    for (int i=0; i<n_h1; i++) {
        fin >> biases_h1[i];
    }
    
    for (int i=0; i<n_h2; i++) {
        fin >> biases_h2[i];
    }
    
    for (int i=0; i<n_output; i++) {
        fin >> biases_output[i];
    }
    fin.close();

    // Recalculate and print outputs for all input patterns
    cout << "Final Outputs:" << endl;
    for (int i = 0; i < 5; ++i) {
        // Forward propagation for each input
        array<double, n_h1> h1_outputs;
        for (int j = 0; j < n_h1; ++j) {
            h1_outputs[j] = 0;
            for (int k = 0; k < n_input; ++k) {
                h1_outputs[j] += static_cast<double>(images[i].pixels[k]) / 255.0 * input_to_h1_weights[k][j];
            }
            h1_outputs[j] += biases_h1[j];  // Add bias
            h1_outputs[j] = sigmoid(h1_outputs[j]);  // Apply activation function
        }

        array<double, n_h2> h2_outputs;
        for (int j = 0; j < n_h2; ++j) {
            h2_outputs[j] = 0;
            for (int k = 0; k < n_h1; ++k) {
                h2_outputs[j] += h1_outputs[k] * h1_to_h2_weights[k][j];
            }
            h2_outputs[j] += biases_h2[j];  // Add bias
            h2_outputs[j] = sigmoid(h2_outputs[j]);  // Apply activation function
        }

        array<double, n_output> output;
        for (int j = 0; j < n_output; ++j) {
            output[j] = 0;
            for (int k = 0; k < n_h2; ++k) {
                output[j] += h2_outputs[k] * h2_to_output_weights[k][j];
            }
            output[j] += biases_output[j];  // Add bias
            output[j] = sigmoid(output[j]);  // Apply activation function
        }

        // Print the input and corresponding output
        printImage(images[i]);
        int max_i = 0;
        for (int j = 0; j < n_output; j++) {
            if (output[j] > output[max_i]) {
                max_i = j;
            }
        }
        cout << "Output: " << max_i << endl;
    }

    return 0;
}
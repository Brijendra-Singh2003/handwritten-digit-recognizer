#include <bits/stdc++.h>
using namespace std;

#include "./src/input.cpp"

const int n_input = 28 * 28;
const int n_h1 = 32;
const int n_h2 = 16;
const int n_output = 10;
const int MAX_ITERATIONS = 20;
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

    // Initialize random number generator for weight initialization
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);  // Random weights in range [-1, 1]

    for (auto &a : input_to_h1_weights)
        for (double& n : a) n = dis(gen);
    for (double& n : biases_h1) n = dis(gen);

    for (auto &a : h1_to_h2_weights)
        for (double& n : a) n = dis(gen);
    for (double& n : biases_h2) n = dis(gen);

    for (auto &a : h2_to_output_weights)
        for (double& n : a) n = dis(gen);
    for (double& n : biases_output) n = dis(gen);

    // Training loop
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        double total_error = 0;

        for (int i = 0; i < images.size(); ++i) {
            // Forward propagation
            // Hidden layer 1
            array<double, n_h1> h1_outputs;
            for (int j = 0; j < n_h1; ++j) {
                h1_outputs[j] = 0;
                for (int k = 0; k < n_input; ++k) {
                    h1_outputs[j] += static_cast<double>(images[i].pixels[k]) / 255.0 * input_to_h1_weights[k][j];
                }
                h1_outputs[j] += biases_h1[j];  // Add bias
                h1_outputs[j] = sigmoid(h1_outputs[j]);  // Apply activation function
            }

            // Hidden layer 2
            array<double, n_h2> h2_outputs;
            for (int j = 0; j < n_h2; ++j) {
                h2_outputs[j] = 0;
                for (int k = 0; k < n_h1; ++k) {
                    h2_outputs[j] += h1_outputs[k] * h1_to_h2_weights[k][j];
                }
                h2_outputs[j] += biases_h2[j];  // Add bias
                h2_outputs[j] = sigmoid(h2_outputs[j]);  // Apply activation function
            }

            // Output layer
            array<double, n_output> output;
            for (int j = 0; j < n_output; ++j) {
                output[j] = 0;
                for (int k = 0; k < n_h2; ++k) {
                    output[j] += h2_outputs[k] * h2_to_output_weights[k][j];
                }
                output[j] += biases_output[j];  // Add bias
                output[j] = sigmoid(output[j]);  // Apply activation function
            }

            // Compute error (Mean Squared Error)
            double error = 0;
            for (int j = 0; j < n_output; ++j) {
                double target = (labels[i] == j) ? 1.0 : 0.0;
                error += 0.1 * pow(target - output[j], 2);
            }
            total_error += error;

            // Backpropagation
            // Output layer
            array<double, n_output> output_deltas;
            for (int j = 0; j < n_output; ++j) {
                double target = (labels[i] == j) ? 1.0 : 0.0;
                output_deltas[j] = (target - output[j]) * sigmoid_derivative(output[j]);
            }

            // Hidden layer
            array<double, n_h2> h2_deltas;
            for (int j = 0; j < n_h2; ++j) {
                h2_deltas[j] = 0;
                for (int k = 0; k < n_output; ++k) {
                    h2_deltas[j] += output_deltas[k] * h2_to_output_weights[j][k];
                }
                h2_deltas[j] *= sigmoid_derivative(h2_outputs[j]);
            }

            // Hidden layer
            array<double, n_h1> h1_deltas;
            for (int j = 0; j < n_h1; ++j) {
                h1_deltas[j] = 0;
                for (int k = 0; k < n_h2; ++k) {
                    h1_deltas[j] += h2_deltas[k] * h1_to_h2_weights[j][k];
                }
                h1_deltas[j] *= sigmoid_derivative(h1_outputs[j]);
            }

            // Update weights for output layer
            for (int j = 0; j < n_output; ++j) {
                biases_output[j] += LEARNING_RATE * output_deltas[j] * 1.0;
                for (int k = 0; k < n_h2; ++k) {
                    h2_to_output_weights[k][j] += LEARNING_RATE * h2_outputs[k] * output_deltas[j];
                }
            }

            // Update weights for h2 layer
            for (int j = 0; j < n_h2; ++j) {
                biases_h2[j] += LEARNING_RATE * h2_deltas[j];
                for (int k = 0; k < n_h1; ++k) {
                    h1_to_h2_weights[k][j] += LEARNING_RATE * h1_outputs[k] * h2_deltas[j];
                }
            }

            // Update weights for h1 layer
            for (int j = 0; j < n_h1; ++j) {
                biases_h1[j] += LEARNING_RATE * h1_deltas[j];
                for (int k = 0; k < n_input; ++k) {
                    input_to_h1_weights[k][j] += LEARNING_RATE * images[i].pixels[k] / 255.0 * h1_deltas[j];
                }
            }
        }

        // Print the total error every 1000 iterations
        // if (iteration % 1000 == 0) {
            cout << "Iteration: " << iteration << ", Total Error: " << total_error << endl;
        // }

        // Stop if the error is sufficiently low
        if (total_error < 1e-6) {
            cout << "Training completed after " << iteration << " iterations!" << endl;
            break;
        }
    }

    // select random 4 indicies
    vector<int> idx(4);
    for (int i=0; i<idx.size(); i++) {
        idx[i] = rand() % images.size();
    }

    // Recalculate and print outputs for all input patterns
    cout << "Final Outputs:" << endl;
    for (int j = 0; j < idx.size(); ++j) {
        int i = idx[j];
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
            cout << output[j] << "  ";
        }
        cout << "Output: " << max_i << endl;
    }

    std::ofstream fout("weights.txt", std::ios::out);

    for (int i=0; i<n_input; i++) {
        for(int j=0; j<n_h1; j++) {
            fout << input_to_h1_weights[i][j] << ' ';
        }
        fout << endl;
    }

    for (int i=0; i<n_h1; i++) {
        for(int j=0; j<n_h2; j++) {
            fout << h1_to_h2_weights[i][j] << ' ';
        }
        fout << endl;
    }

    for (int i=0; i<n_h2; i++) {
        for(int j=0; j<n_output; j++) {
            fout << h2_to_output_weights[i][j] << ' ';
        }
        fout << endl;
    }
    
    for (int i=0; i<n_h1; i++) {
        fout << biases_h1[i] << ' ';
        fout << endl;
    }
    
    for (int i=0; i<n_h2; i++) {
        fout << biases_h2[i] << ' ';
        fout << endl;
    }
    
    for (int i=0; i<n_output; i++) {
        fout << biases_output[i] << ' ';
        fout << endl;
    }

    return 0;
}
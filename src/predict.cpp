#include "./input.cpp"

class mnist_model {
private:
    const int n_input = 28 * 28;
    const int n_h1 = 32;
    const int n_h2 = 16;
    const int n_output = 10;

    std::vector<std::vector<double>> input_to_h1_weights;
    std::vector<std::vector<double>> h1_to_h2_weights;
    vector<vector<double>> h2_to_output_weights;

    vector<double> biases_h1;
    vector<double> biases_h2;
    vector<double> biases_output;

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }
    
public:
    mnist_model(string wt_file_path) {
        input_to_h1_weights = std::vector<std::vector<double>>(n_input, std::vector<double>(n_h1));
        h1_to_h2_weights = vector<vector<double>>(n_h1, vector<double>(n_h2));
        h2_to_output_weights = vector<vector<double>>(n_h2, vector<double>(n_output));
    
        biases_h1 = vector<double>(n_h1);
        biases_h2 = vector<double>(n_h2);
        biases_output = vector<double>(n_output);

        std::ifstream fin(wt_file_path, std::ios::in);

        for (int i = 0; i < n_input; i++) {
            for (int j = 0; j < n_h1; j++) {
                fin >> input_to_h1_weights[i][j];
            }
        }

        for (int i = 0; i < n_h1; i++) {
            for (int j = 0; j < n_h2; j++) {
                fin >> h1_to_h2_weights[i][j];
            }
        }

        for (int i = 0; i < n_h2; i++) {
            for (int j = 0; j < n_output; j++) {
                fin >> h2_to_output_weights[i][j];
            }
        }

        for (int i = 0; i < n_h1; i++) {
            fin >> biases_h1[i];
        }

        for (int i = 0; i < n_h2; i++) {
            fin >> biases_h2[i];
        }

        for (int i = 0; i < n_output; i++) {
            fin >> biases_output[i];
        }
    }

    int predict(MNISTImage image) {
        // Forward propagation for each input
        vector<double> h1_outputs(n_h1);
        for (int j = 0; j < n_h1; ++j) {
            h1_outputs[j] = 0;
            for (int k = 0; k < n_input; ++k) {
                h1_outputs[j] += static_cast<double>(image.pixels[k]) / 255.0 * input_to_h1_weights[k][j];
            }
            h1_outputs[j] += biases_h1[j];  // Add bias
            h1_outputs[j] = sigmoid(h1_outputs[j]);  // Apply activation function
        }

        vector<double> h2_outputs(n_h2);
        for (int j = 0; j < n_h2; ++j) {
            h2_outputs[j] = 0;
            for (int k = 0; k < n_h1; ++k) {
                h2_outputs[j] += h1_outputs[k] * h1_to_h2_weights[k][j];
            }
            h2_outputs[j] += biases_h2[j];  // Add bias
            h2_outputs[j] = sigmoid(h2_outputs[j]);  // Apply activation function
        }

        vector<double> output(n_output);
        for (int j = 0; j < n_output; ++j) {
            output[j] = 0;
            for (int k = 0; k < n_h2; ++k) {
                output[j] += h2_outputs[k] * h2_to_output_weights[k][j];
            }
            output[j] += biases_output[j];  // Add bias
            output[j] = sigmoid(output[j]);  // Apply activation function
        }

        // Print the input and corresponding output
        printImage(image);
        int max_i = 0;
        for (int j = 0; j < n_output; j++) {
            if (output[j] > output[max_i]) {
                max_i = j;
            }
            cout << j << ": " << output[j] << ",  ";
        }
        cout << "\nOutput: " << max_i << endl;
        return max_i;
    }
};
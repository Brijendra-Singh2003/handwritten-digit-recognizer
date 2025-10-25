#include <bits/stdc++.h>
using namespace std;

struct MNISTImage {
    std::vector<uint8_t> pixels;
    int rows;
    int cols;
};

// Function to read IDX3-UBYTE file
std::vector<MNISTImage> readIDX3Ubyte(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    // Read the magic number (4 bytes)
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);  // Convert from big-endian

    if (magic_number != 2051) {  // 2051 is the magic number for image files
        std::cerr << "Error: Invalid magic number (" << magic_number << ")" << std::endl;
        return {};
    }

    // Read the number of images, rows, and columns (each 4 bytes)
    uint32_t num_images = 0, num_rows = 0, num_cols = 0;
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    std::cout << "Reading " << num_images << " images of size " 
              << num_rows << "x" << num_cols << std::endl;

    // Read image data
    std::vector<MNISTImage> images(num_images);
    for (uint32_t i = 0; i < num_images; i++) {
        MNISTImage img;
        img.pixels.resize(num_rows * num_cols);
        img.rows = num_rows;
        img.cols = num_cols;
        file.read(reinterpret_cast<char*>(img.pixels.data()), num_rows * num_cols);
        images[i] = img;
    }

    file.close();
    return images;
}

// Function to read IDX1-UBYTE file
std::vector<uint8_t> readIDX1Ubyte(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    // Read the magic number (4 bytes)
    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);  // Convert from big-endian

    if (magic_number != 2049) {  // 2049 is the magic number for label files
        std::cerr << "Error: Invalid magic number (" << magic_number << ")" << std::endl;
        return {};
    }

    // Read the number of labels (4 bytes)
    uint32_t num_labels = 0;
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = __builtin_bswap32(num_labels);  // Convert from big-endian

    std::cout << "Reading " << num_labels << " labels" << std::endl;

    // Read label data
    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    file.close();
    return labels;
}

char bt[] = {' ','-','+','?','/','T','H','#','@','$','&'};

// Function to print an image as ASCII (for visualization)
void printImage(const MNISTImage& img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int pixel = img.pixels[i * img.cols + j];
            std::cout << bt[pixel/25] << ' ';
        }
        std::cout << std::endl;
    }
}


#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>

#define IMAGE_SIZE 784
#define MAX_IMAGES 60000

uint32_t readUInt32(std::ifstream& stream) {
    uint8_t bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Returns pointer to [numImages x IMAGE_SIZE] array of ints
int* loadMNISTImages(const std::string& filename, int& numImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open image file.");

    uint32_t magic = readUInt32(file);
    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file.");

    numImages = readUInt32(file);
    if (numImages > MAX_IMAGES) numImages = MAX_IMAGES;

    uint32_t rows = readUInt32(file);
    uint32_t cols = readUInt32(file);

    uint8_t* buffer = static_cast<uint8_t*>(malloc(numImages * IMAGE_SIZE));
    int* images = static_cast<int*>(malloc(sizeof(int) * numImages * IMAGE_SIZE));

    if (!buffer || !images) throw std::runtime_error("Memory allocation failed.");

    file.read(reinterpret_cast<char*>(buffer), numImages * IMAGE_SIZE);

    for (int i = 0; i < numImages * IMAGE_SIZE; ++i) {
        images[i] = static_cast<int>(buffer[i]);
    }

    free(buffer);
    return images;
}

// Returns pointer to int[numLabels]
int* loadMNISTLabels(const std::string& filename, int& numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open label file.");

    uint32_t magic = readUInt32(file);
    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file.");

    numLabels = readUInt32(file);
    if (numLabels > MAX_IMAGES) numLabels = MAX_IMAGES;

    uint8_t* buffer = static_cast<uint8_t*>(malloc(numLabels));
    int* labels = static_cast<int*>(malloc(sizeof(int) * numLabels));

    if (!buffer || !labels) throw std::runtime_error("Memory allocation failed.");

    file.read(reinterpret_cast<char*>(buffer), numLabels);

    for (int i = 0; i < numLabels; ++i) {
        labels[i] = static_cast<int>(buffer[i]);
    }

    free(buffer);
    return labels;
}


int main() {
    int numImages = 0, numLabels = 0;
    int* images = loadMNISTImages("train-images-idx3-ubyte", numImages);
    int* labels = loadMNISTLabels("train-labels-idx1-ubyte", numLabels);

    std::cout << "Loaded " << numImages << " images.\n";
    std::cout << "Label of first image: " << labels[0] << "\n";

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            int pixel = images[0 * IMAGE_SIZE + i * 28 + j];
            std::cout << (pixel > 128 ? '#' : ' ');
        }
        std::cout << "\n";
    }

    free(images);
    free(labels);
    return 0;
}

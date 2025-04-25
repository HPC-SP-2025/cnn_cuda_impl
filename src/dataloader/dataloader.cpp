# include <iostream>
# include <vector>
# include <string>
#include "../../include/cnn_library/dataloader/dataloader.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

uint32_t readUInt32(std::ifstream& stream) 
{
    uint8_t bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<std::vector<uint8_t>> readMNISTImages(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open image file.");

    uint32_t magic = readUInt32(file);
    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file.");

    uint32_t numImages = readUInt32(file);
    uint32_t numRows = readUInt32(file);
    uint32_t numCols = readUInt32(file);

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(numRows * numCols));

    for (auto& img : images) {
        file.read(reinterpret_cast<char*>(img.data()), img.size());
    }

    return images;
}

std::vector<uint8_t> readMNISTLabels(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open label file.");

    uint32_t magic = readUInt32(file);
    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file.");

    uint32_t numLabels = readUInt32(file);
    std::vector<uint8_t> labels(numLabels);
    file.read(reinterpret_cast<char*>(labels.data()), labels.size());

    return labels;
}

# include <fstream>

DataLoader::DataLoader(std::string data_path, int batch_size)
{
    this->data_path = data_path;
    this->batch_size = batch_size;

    // Load image paths and labels from the text file
    std::ifstream file(data_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << data_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find_last_of(' ');
        if (pos != std::string::npos) {
            image_paths.push_back(line.substr(0, pos));
            labels.push_back(std::stoi(line.substr(pos + 1)));
            num_samples++;
        }
    }
    file.close();

    num_batches = (num_samples + batch_size - 1) / batch_size; // Calculate number of batches
}



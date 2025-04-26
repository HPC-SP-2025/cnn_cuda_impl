#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cstdlib>  // for rand, srand
#include <ctime>    // for time
#include <algorithm>
#include "../../include/cnn_library/dataloader/dataloader.h"


using namespace std;
#define IMAGE_SIZE 784
#define MAX_IMAGES 60000

// int get_random_index(int start, int end) {
//     if (start > end) {
//         throw std::invalid_argument("Start cannot be greater than end.");
//     }
//     static std::random_device rd;
//     static std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(start, end);
//     return dis(gen);
// }


std::vector<int> get_random_indices(int total_images, int batch_size) {
    std::vector<int> indices;

    // Fill with 0 to total_images-1
    for (int i = 0; i < total_images; ++i)
        indices.push_back(i);

    // Shuffle using basic rand()
    std::srand(std::time(nullptr));
    std::random_shuffle(indices.begin(), indices.end());

    // Take first batch_size elements
    indices.resize(batch_size);
    return indices;
}

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


DataLoader::DataLoader(std::string data_path, std::string label_path, int batch_size, int number_of_images)
{
    this->numImages = number_of_images;
    this->batch_size = batch_size;
    this->data_path = data_path;
    this->label_path = label_path;

    // Load images and labels
    this->images = loadMNISTImages(data_path, numImages);
    this->labels = loadMNISTLabels(label_path, numImages);
}


Batch DataLoader::load_batch(int idx)
{
    // Allocate memory for the batch
    float* batch = static_cast<float*>(malloc(sizeof(float) * batch_size * IMAGE_SIZE));
    float* labels_batch = static_cast<float*>(malloc(sizeof(float) * batch_size));

    // Error Handling
    if (!labels_batch) throw std::runtime_error("Memory allocation failed for labels batch.");
    if (!batch) throw std::runtime_error("Memory allocation failed for batch.");


    // FOR MULTIPLE BATCH SIZE
    // for (int i = 0; i < batch_size; ++i) 
    // {
    //     for (int j = 0; j < IMAGE_SIZE; ++j) 
    //     {
    //         batch[i * IMAGE_SIZE + j] = static_cast<float>(images[idx + (i * IMAGE_SIZE) + j]);
    //     }
    //     labels_batch[i] = static_cast<float>(labels[idx+i]);
    // }

    // FOR SINGLE BATCH SIZE
    for (int j = 0; j < IMAGE_SIZE; ++j) 
    {
        batch[j] = static_cast<float>(images[idx * IMAGE_SIZE + j]);
    }
    labels_batch[0] = static_cast<float>(labels[idx]);

    return Batch{batch, labels_batch};

}

DataLoader::~DataLoader() {
    free(images);
    free(labels);
}






// FOR TESTING THE DATALODER
// int main() {
//     try {
//         string image_path = "/home/MORGRIDGE/akazi/HPC_Assignments/Final_Project/CNN_Implementation_on_CUDA/MNIST_Dataset/train-images-idx3-ubyte";
//         string label_path = "/home/MORGRIDGE/akazi/HPC_Assignments/Final_Project/CNN_Implementation_on_CUDA/MNIST_Dataset/train-labels-idx1-ubyte";
        
//         int batch_size = 50;
//         int image_size = 28 * 28;
//         DataLoader* dataloader = new DataLoader(image_path, label_path, batch_size, 1000);

//         Batch batch  = dataloader->load_batch();
//         float* image_batch = batch.images;
//         float* labels = batch.labels;

//         for (int k = 0; k < batch_size; ++k) 
//         {
//             std::cout << "Label: " << labels[k] << "\n";
//             for (int i = 0; i < 28; ++i) {
//                 for (int j = 0; j < 28; ++j) {
//                     int pixel = image_batch[k * image_size + i * 28 + j];
//                     std::cout << (pixel > 128 ? '#' : ' ');
//                 }
//                 std::cout << "\n";
//             }
//         }

//         free(image_batch);
//     } 
    
//         catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }


# include <iostream>
# include <vector>
# include <string>

using namespace std;

struct Batch {
    float* images;
    float* labels;
};

class DataLoader
{
    private:
        std::string data_path;
        std::string label_path;
        std::vector<std::string> image_paths;
        int batch_size;
        int numImages;
        int* images;
        int* labels;
 

    public:

        DataLoader(std::string data_path, std::string label_path, int batch_size, int number_of_images);
        
        ~DataLoader();

        Batch load_batch(int idx);      
        
        float* read_image(string image_path);
};

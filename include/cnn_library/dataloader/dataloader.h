# include <iostream>
# include <vector>
# include <string>

using namespace std;

class DataLoader
{
    private:
        std::string data_path;
        std::vector<std::string> image_paths;
        std::vector<int> labels;
        int batch_size;
        int current_index;
        int num_classes;
        std::vector<cv::Mat> batch_images;
        std::vector<int> batch_labels;
        int num_samples;
        int current_batch_index;
        int num_batches;


    public:

        DataLoader(std::string data_path, int batch_size);
        
        ~DataLoader();

        float* load_batch();      
        
        float* read_image(string image_path);

        

}
# include <iostream>
# include <vector>
# include <string>

class DataLoader
{
public:
    DataLoader(const std::string& data_path, int batch_size)
        : data_path(data_path), batch_size(batch_size)
    {
        // Load the dataset
        load_data();
    }

    void load_data()
    {
        // Load the dataset from the specified path
        std::cout << "Loading data from " << data_path << " with batch size " << batch_size << std::endl;
        // Implement the logic to load the dataset here
    }
}
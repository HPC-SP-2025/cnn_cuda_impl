# include <iostream>

int main(int argc, char* argv[])
{ 

    // Take epochs and device as command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <epochs> <device>" << std::endl;
        return 1;
    }

    int epochs = std::stoi(argv[1]);
    std::string device = argv[2];

    
    
    // Create the training for loop
    std::cout << "Training on " << device << " for " << epochs << " epochs." << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        // Create a random image
        std::vector<std::vector<int>> image(28, std::vector<int>(28, rand() % 256));
        
        // Create a random label
        int label = rand() % 10;

        for (int epoch = 0; epoch < epochs; i++)
        {
            for (int batch_index = 0; batch_index < number_of_batches; batch_index++)
            {
                // Forwardfeed the model
                prediction = net.forward();

                // Backpropagation
                net.backward(prediction, label_vector);

                // Update the weights
                net.update_weights();

                // Print the loss
                int loss = net.loss(prediction, label_vector);
                std::cout << "Epoch: " << epoch << ", Batch: " << batch_index << ", Loss: " << loss << std::endl;

            }

        }
    }
}
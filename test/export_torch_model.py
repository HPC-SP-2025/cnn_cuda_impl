import torch
import torch.nn as nn


if __name__ == "__main__":


    # Example of creating a simple linear layer
    # This is just a placeholder for the actual test
    # In a real test, you would use assert statements to check the output
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear_layer1 = nn.Linear(in_features=700, out_features=1024, dtype=torch.float32)
            self.relu1 = nn.ReLU()
            self.linear_layer2 = nn.Linear(in_features=1024, out_features=256, dtype=torch.float32)
            self.relu2 = nn.ReLU()
            self.linear_layer3 = nn.Linear(in_features=256, out_features=10, dtype=torch.float32)    
    

        def forward(self, x):
            x = self.linear_layer1(x.to(dtype=torch.float32))
            x = self.relu1(x)
            x = self.linear_layer2(x)
            x = self.relu2(x)
            x = self.linear_layer3(x)
            x = torch.softmax(x, dim=-1)

            return x

    # Example usage of the model
    model = SimpleModel()
    
    # Open a file to write the model weights and biases
    with open("model_weights.txt", "w") as f:
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                weights = layer.weight.T.flatten().tolist()
                biases = layer.bias.flatten().tolist()
                combined = weights + biases
                combined = " ".join(map(str, combined))
                f.write(combined)
                f.write("\n")
            else:
                f.write("\n")




    # Example input tensor
    input_tensor = torch.tensor([[10.0] * 700])
    output = model(input_tensor)
    # Compute Cross-Entropy Loss
    target = torch.tensor([0])  # Example target class index
    cce_loss = nn.CrossEntropyLoss()
    loss = cce_loss(output, target)
    print("Cross-Entropy Loss:", loss.item())
    print("Output:", output)

    




            
        
    

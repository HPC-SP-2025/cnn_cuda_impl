import matplotlib.pyplot as plt


def save_loss_plot(iterations, loss_values, output_path='loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, loss_values, label='Loss', color='blue', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

# Path to your text file
file_path = 'build/iteration_wise_loss_values.txt'

# Read the loss values from the file
with open(file_path, 'r') as file:
    loss_values = [float(line.strip()) for line in file if line.strip()]

# Create an x-axis representing iteration indices
iterations = list(range(1, len(loss_values) + 1))
iterations = [i * 128 for i in iterations]  # Assuming each iteration corresponds to 100 iterations in the training process

# Plot the graph


# Call the function to save the plot
save_loss_plot(iterations, loss_values, output_path='loss_plot_gpu.png')

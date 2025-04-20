#!/usr/bin/env zsh

#SBATCH --job-name=cnn_test
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=cnn_test.out
#SBATCH --error=cnn_test.err
#SBATCH --time=0-00:10:00

cd $SLURM_SUBMIT_DIR

# load the gcc for compiling C++ programs
module load gcc/11.3.0

# load the nvcc for compiling cuda programs
module load nvidia/cuda/11.8.0

nvcc test_relu.cu ../src/layers/relu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o relu

input_size=8
output_size=8
threads_per_block=1024
device=0
for i in {0..0}
do
    ./relu $input_size $output_size $threads_per_block $device
done
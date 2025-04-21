#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -c 1
#SBATCH -G 1 --gpus-per-task=1
#SBATCH -J loss_test 
#SBATCH -o loss_test.out -e loss_test.err


# load the nvcc for compiling cuda programs
module load nvidia/cuda/11.8.0

nvcc test_loss.cu ../src/layers/cross_entropy_loss.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o test_loss

./test_loss
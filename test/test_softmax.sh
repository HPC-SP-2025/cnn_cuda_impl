#!/usr/bin/env zsh

#SBATCH --job-name=softmax_test
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=softmax_test.out
#SBATCH --error=softmax_test.err
#SBATCH --time=0-00:10:00

cd $SLURM_SUBMIT_DIR

# load the gcc for compiling C++ programs
module load gcc/11.3.0

# load the nvcc for compiling cuda programs
module load nvidia/cuda/11.8.0

nvcc test_softmax.cu ../src/layers/softmax.cu -Xcompiler -O3 -Xcompiler -Wzero-as-null-pointer-constant -Xptxas -O3 -std=c++17 -o softmax

num_classes=5
batch_size=1
device=0
./softmax $num_classes $batch_size $device

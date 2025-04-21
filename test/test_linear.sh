#!/usr/bin/env zsh

#SBATCH --job-name=linear_test
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=linear_test.out
#SBATCH --error=linear_test.err
#SBATCH --time=0-00:10:00

cd $SLURM_SUBMIT_DIR

# load the gcc for compiling C++ programs
module load gcc/11.3.0

# load the nvcc for compiling cuda programs
module load nvidia/cuda/11.8.0

nvcc test_linear1.cu ../src/layers/linear.cu -Xcompiler -O3 -Xcompiler -Wzero-as-null-pointer-constant -Xptxas -O3 -std=c++17 -o linear

./linear 10 5 1 0

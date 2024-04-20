CXX		=g++
GPU		=-O3 --gpu-architecture=compute_80 --use_fast_math --gpu-code=sm_80 -Xptxas -v

gpu: 
	nvcc -std=c++17 -I include/ $(GPU) softmax_cuda.cu -o softmax_cuda.bin $(PROFILE) $(DEBUG)

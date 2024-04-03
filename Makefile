CXX ?= g++
cpu:
	$(CXX) -std=c++17 -I include/ -march=native -fopenmp softmax_cpu.cpp -o softmax_cpu.bin -ffast-math -O3

gpu: 
	nvcc -std=c++17 -I include/ -O3 --gpu-architecture=compute_80 --gpu-code=sm_80 -Xptxas -v softmax_cuda.cu -o softmax_cuda.bin $(PROFILE) $(DEBUG)

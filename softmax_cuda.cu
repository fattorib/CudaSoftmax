#include "common.hpp"
#include "kernels/softmax.cuh"
#include "softmax_cpu.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#define CHECK_TRUE(v) (assert(v == true))

int main(void) {
  const long numRows = 4096;

  const long numCols = 16384;

  long warmup_iters, benchmark_iters;

#ifdef PROFILE
  warmup_iters = 1;
  benchmark_iters = 1;

#else
  warmup_iters = 25;
  benchmark_iters = 1000;

#endif

  float *d_in, *d_out;

  // make host-side arrays
  std::vector<float> in(numRows * numCols);
  std::vector<float> out(numRows * numCols);

  fill_matrix(in);

  // allocate memory for GPU arrays
  cudaMalloc(&d_in, in.size() * sizeof(float));
  cudaMalloc(&d_out, out.size() * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // copy from host to GPU
  cudaMemcpy(d_in, in.data(), in.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  const int numThread = 128; // makes sense to think about this as # of warps
                             // since we are doing warp reductions

  dim3 blockDim(numThread);

  assert(numCols % (4 * numThread) == 0);

  // launch one threadblock per row
  dim3 gridDim(numRows);

  int bytesBlock = sizeof(float) * numCols;

  cudaFuncSetAttribute(softmax_kernel<numCols, numThread>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytesBlock);

  std::cout << "Warmup started" << std::endl;
  for (int i = 0; i < warmup_iters; i++) {
    softmax_kernel<numCols, numThread>
        <<<gridDim, blockDim, bytesBlock>>>(d_in, d_out);
  }

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  std::cout << "Benchmark started" << std::endl;
  cudaEventRecord(start);
  for (int i = 0; i < benchmark_iters; i++) {
    softmax_kernel<numCols, numThread>
        <<<gridDim, blockDim, bytesBlock>>>(d_in, d_out);
  }

  cudaEventRecord(stop);

  cudaMemcpy(out.data(), d_out, out.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  double gb = benchmark_iters * 2 * sizeof(float) * numRows * numCols * 1e-9;
  double elapsed_time = double(milliseconds) * 1e-3;

  printf("Total elapsed time: (%7.6f) s, performance: (%7.1f) GB/s, memory "
         "reads & writes (GB): (%6.1lf) \n\n",
         elapsed_time, (gb) / elapsed_time, gb);

  std::vector<float> cpuOut(numRows * numCols);

  softmaxCpu<numRows, numCols>(in, cpuOut);

  // compute errors
  printf("Error checking:\n");
  printf("Relative Error (%7.8f) \n", relative_error(cpuOut, out));
}
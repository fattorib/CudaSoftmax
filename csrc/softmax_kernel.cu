#include <torch/types.h>
#include "include/kernels/softmax.cuh"


torch::Tensor fusedSoftmax(torch::Tensor in) {
  const int nRow = in.size(0);
  const int nCol = in.size(1);

  auto out = torch::empty_like(in);

  dim3 gridDim(nRow);

  int sramBlock = sizeof(float) * nCol;

  dim3 blockDim;

  switch (nCol) {

  // TODO: Support for masked indices
  // TODO: Better support for templates

  case 1024:
    blockDim.x = 32;
    cudaFuncSetAttribute(softmax_kernel<1024, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<1024, 32><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  case 2048:
    blockDim.x = 32;
    cudaFuncSetAttribute(softmax_kernel<2048, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<2048, 32><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  case 2560:
    blockDim.x = 32;
    cudaFuncSetAttribute(softmax_kernel<2560, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<2560, 32><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  case 3072:
    blockDim.x = 32;
    cudaFuncSetAttribute(softmax_kernel<3072, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<3072, 32><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  case 4096:
    blockDim.x = 32;
    cudaFuncSetAttribute(softmax_kernel<4096, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<4096, 32><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;
  
  case 5120:
    blockDim.x = 32;
    cudaFuncSetAttribute(softmax_kernel<5120, 32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<5120, 32><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;
  
  case 7680:
    blockDim.x = 64;
    cudaFuncSetAttribute(softmax_kernel<7680, 64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<7680, 64><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  case 8192:
    blockDim.x = 64;
    cudaFuncSetAttribute(softmax_kernel<8192, 64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<8192, 64><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  case 16384:
    blockDim.x = 128;
    cudaFuncSetAttribute(softmax_kernel<16384, 128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sramBlock);
    softmax_kernel<16384, 128><<<gridDim, blockDim, sramBlock>>>(
        in.data_ptr<float>(), out.data_ptr<float>());
    break;

  default:
    throw std::runtime_error("Unsupported value for dimension 1 of tensor.");
  }

  return out;
}
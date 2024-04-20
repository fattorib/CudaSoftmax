#include <cuda.h>
#include <cuda_runtime.h>

#define FLOAT4 4
#define WARP 32

// when converted to binary this is used as a 32-element bitmask
// in binary this is 11111111111111111111111111111111 (all threads in warp)
#define FULL_MASK 0xffffffff

// atomicMax not impl in CUDA for floats
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ static float atomicMax(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

template <int nCol, int numThread>
__global__ void softmax_kernel(float *in, float *out) {

  const u_int rowIndex = blockIdx.x;
  const u_int rowStride = nCol;

  // update ptr starts to correct row
  in += rowStride * rowIndex;
  out += rowStride * rowIndex;

  extern __shared__ float smem[];
  float *rowShared = &smem[0];          // stores the row
  float *reductionShared = &smem[nCol]; // stores a single float required for
                                        // reductions across threads

  reductionShared[0] = -1 * INFINITY; // need to pre-populate comparison value
  // in SMEM before using max atomic

  const u_int tx = threadIdx.x;
  const u_int laneIdx = threadIdx.x % WARP;

  // number of float4 loads to be issued per thread
  const int loadPerThread = (nCol / numThread / FLOAT4);

  // number of registers to compute per thread
  const int compPerThread = (nCol / numThread);

  int loadWidth = numThread * FLOAT4; // total width of loads per-TB

  for (int loadIdx = 0; loadIdx < loadPerThread; loadIdx++) {
    float4 tmp =
        reinterpret_cast<float4 *>(&in[FLOAT4 * tx + (loadWidth * loadIdx)])[0];
    rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 0] = tmp.x;
    rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 1] = tmp.y;
    rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 2] = tmp.z;
    rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 3] = tmp.w;
  }

  float reg[compPerThread];
  float acc = -1 * INFINITY;
  __syncthreads();

#pragma unroll
  for (int idx = 0; idx < compPerThread; idx++) {
    reg[idx] = rowShared[compPerThread * tx + idx];
  }

#pragma unroll
  for (int idx = 0; idx < compPerThread; idx++) {
    acc = fmaxf(acc, reg[idx]);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    acc = fmaxf(acc, __shfl_down_sync(FULL_MASK, acc, offset));
  }

  __syncthreads();

  if (laneIdx == 0) {
    atomicMax(reductionShared, acc);
  }
  __syncthreads();

  float rowMax = reductionShared[0];
  __syncthreads();

  // clear SMEM -> all threads write this value
  reductionShared[0] = 0.0f;
  acc = 0.0f;

  for (int idx = 0; idx < compPerThread; idx++) {
    reg[idx] = expf(reg[idx] - rowMax);
    acc += reg[idx];
  }

  // reduce across warp
  for (int offset = 16; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(FULL_MASK, acc, offset);
  }

  __syncthreads();

  if (laneIdx == 0) {
    atomicAdd(reductionShared, acc);
  }

  __syncthreads();

  float denom = reductionShared[0];

  // write back to SMEM
  for (int idx = 0; idx < compPerThread; idx++) {
    rowShared[compPerThread * tx + idx] = reg[idx] / denom;
  }

  __syncthreads();

  for (int loadIdx = 0; loadIdx < loadPerThread; loadIdx++) {
    float4 tmp;
    tmp.x = rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 0];
    tmp.y = rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 1];
    tmp.z = rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 2];
    tmp.w = rowShared[FLOAT4 * tx + (loadWidth * loadIdx) + 3];
    reinterpret_cast<float4 *>(&out[FLOAT4 * tx + (loadWidth * loadIdx)])[0] =
        tmp;
  }
}
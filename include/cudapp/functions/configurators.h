//
// Created by Daniel Simon on 3/29/20.
//

#ifndef CUDAPP_CONFIGURATORS_H
#define CUDAPP_CONFIGURATORS_H

#include <utility>

#include <cuda_runtime_api.h>

#include "cudapp/exceptions/cuda_exception.h"

namespace cudapp {

template <typename ... Args>
__host__ __device__ inline cudaFuncAttributes KernelGetAttributes(void(*function)(Args ...)) {
  cudaFuncAttributes out{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cudaError_t ret = cudaFuncGetAttributes(&out, function);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
  return out;
}

template <typename ... Args>
__host__ inline void KernelSetMaxDynamicSharedMemorySize(void(*function)(Args ...), int value) {
  cudaError_t ret = cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, value);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
}

template <typename ... Args>
__host__ inline void KernelSetPreferredSharedMemoryCarveout(void(*function)(Args ...), int value) {
  cudaError_t ret = cudaFuncSetAttribute(function, cudaFuncAttributePreferredSharedMemoryCarveout, value);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
}

enum struct CachePreference {
  None=cudaFuncCachePreferNone,
  Shared=cudaFuncCachePreferShared,
  L1=cudaFuncCachePreferL1,
  Equal=cudaFuncCachePreferShared,
};

template <typename ... Args>
__host__ inline void KernelSetCacheConfig(void(*function)(Args ...), CachePreference preference) {
  cudaError_t ret = cudaFuncSetCacheConfig(function, preference);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
}

enum struct SharedMemBankSize {
  Default=cudaSharedMemBankSizeDefault,
  FourByte=cudaSharedMemBankSizeFourByte,
  EightByte=cudaSharedMemBankSizeEightByte,
};

template <typename ... Args>
__host__ inline void KernelSetSharedMemConfig(void(*function)(Args ...), SharedMemBankSize size) {
  cudaError_t ret = cudaFuncSetSharedMemConfig(function, size);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
}

template <typename ... Args>
__host__ inline int SupportedSimultaneousBlocks(void(*function)(Args ...), dim3 block_size,
    std::size_t shared_memory_size, bool caching_override=false) {
  return SupportedSimultaneousBlocks(function, block_size.x * block_size.y * block_size.z,
      shared_memory_size, caching_override);
}
template <typename ... Args>
__host__ inline int SupportedSimultaneousBlocks(void(*function)(Args ...), int block_size,
    std::size_t shared_memory_size, bool caching_override=false) {
  int out;
  cudaError_t ret = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&out,
      reinterpret_cast<const void*>(function), block_size, shared_memory_size,
      caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
  return out;
}

struct OptimalLaunch {
  int grid_size;
  int block_size;
};

template <typename ... Args>
__host__ inline OptimalLaunch KernelAutomaticLaunchSizes(void(*function)(Args ...),
    std::size_t shared_memory=0, int block_size_limit=0, bool caching_override=false) {
  return KernelAutomaticLaunchSizes(function, [shared_memory](int)->std::size_t{return shared_memory;},
      block_size_limit, caching_override);
}
template <typename ... Args>
__host__ inline OptimalLaunch KernelAutomaticLaunchSizes(void(*function)(Args ...),
    std::size_t(*memory_calculator)(int), int block_size_limit=0, bool caching_override=false) {
  OptimalLaunch out{1, 1};
  cudaError_t ret = cudaOccupancyMaxPotentialBlockSizeVariableSMem(&out.grid_size,
      &out.block_size, function, memory_calculator, block_size_limit,
      caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
  return out;
}

}

#endif //CUDAPP_CONFIGURATORS_H

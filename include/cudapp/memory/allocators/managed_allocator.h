//
// Created by Daniel Simon on 2/5/20.
//

#ifndef CUDAPP_MANAGED_ALLOCATOR_H
#define CUDAPP_MANAGED_ALLOCATOR_H

#include "cudapp/utilities/ide_helpers.h"

#include <new>
#include <utility>

#include "cudapp/exceptions/cuda_exception.h"

namespace cudapp {

template <typename T>
class DeviceAllocator;

template <typename T>
class ManagedAllocator {
 public:
  typedef T value_type;
  ManagedAllocator() = default;

  template <typename U>
  constexpr explicit ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) noexcept(false) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    T* pointer = nullptr;
    cudaError_t ret = cudaMallocManaged(reinterpret_cast<void**>(&pointer), n * sizeof(T));
    if (ret == cudaSuccess) {
      return pointer;
    } else {
      throw CudaException(ret);
    }
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    cudaError_t ret = cudaFree(pointer);
    if (ret != cudaSuccess) {
      // Uncatchable exception
      throw CudaException(ret);
    }
  }
};

template <class T, class U>
__host__ __device__ constexpr bool operator==(const ManagedAllocator<T>&, const ManagedAllocator<U>&) noexcept {
  return true;
}

template <class T, class U>
__host__ __device__ constexpr bool operator==(const ManagedAllocator<T>&, const DeviceAllocator<U>&) noexcept {
  return true;
}

template <class T, class U>
__host__ __device__ constexpr bool operator!=(const ManagedAllocator<T>&, const ManagedAllocator<U>&) noexcept {
  return false;
}

template <class T, class U>
__host__ __device__ constexpr bool operator!=(const ManagedAllocator<T>&, const DeviceAllocator<U>&) noexcept {
  return false;
}


}

#endif //CUDAPP_MANAGED_ALLOCATOR_H

//
// Created by Daniel Simon on 2/5/20.
//

#ifndef CUDAPP_PINNED_ALLOCATOR_H
#define CUDAPP_PINNED_ALLOCATOR_H

#include "include/cudapp/utilities/ide_helpers.h"

#include <new>
#include <utility>

#include "include/cudapp/exceptions/cuda_exception.h"

namespace cudapp {

template <typename T>
class PinnedAllocator {
 public:
  typedef T value_type;
  constexpr PinnedAllocator() noexcept = default;

  template <typename U>
  constexpr explicit PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) noexcept(false) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    T* pointer = nullptr;
    cudaError_t ret = cudaMallocHost(reinterpret_cast<void**>(&pointer), n * sizeof(T));
    if (ret == cudaSuccess) {
      return pointer;
    } else {
      throw CudaException(ret);
    }
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    cudaError_t ret = cudaFreeHost(pointer);
    if (ret != cudaSuccess) {
      // Uncatchable exception
      throw CudaException(ret);
    }
  }
};

template <class T, class U>
__host__ __device__ constexpr bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<U>&) {
  return true;
}

template <class T, class U>
__host__ __device__ constexpr bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<U>&) {
  return false;
}


}

#endif //CUDAPP_PINNED_ALLOCATOR_H

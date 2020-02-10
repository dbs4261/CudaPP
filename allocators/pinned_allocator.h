//
// Created by Daniel Simon on 2/5/20.
//

#ifndef CUDAPP_PINNED_ALLOCATOR_H
#define CUDAPP_PINNED_ALLOCATOR_H

#include "utilities/ide_helpers.h"

#include <new>
#include <utility>

namespace cudapp {

template <typename T>
class PinnedAllocator {
 public:
  typedef T value_type;
  PinnedAllocator() = default;

  template <typename U>
  constexpr PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    T* pointer = nullptr;
    if (cudaMallocHost(reinterpret_cast<void**>(&pointer), n * sizeof(T)) == cudaSuccess) {
      return pointer;
    }
    throw std::bad_alloc();
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    cudaFreeHost(pointer);
  }
};

template <class T, class U>
__host__ __device__ bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<U>&) {
  return true;
}

template <class T, class U>
__host__ __device__ bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<U>&) {
  return false;
}


}

#endif //CUDAPP_PINNED_ALLOCATOR_H

//
// Created by developer on 2/5/20.
//

#ifndef CUDAPP_MANAGED_ALLOCATOR_H
#define CUDAPP_MANAGED_ALLOCATOR_H

#include "utilities/ide_helpers.h"

#include <new>
#include <utility>

namespace cudapp {

template <typename T>
class DeviceAllocator;

template <typename T>
class ManagedAllocator {
 public:
  typedef T value_type;
  ManagedAllocator() = default;

  template <typename U>
  constexpr ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    T* pointer = nullptr;
    if (cudaMallocManaged(reinterpret_cast<void**>(&pointer), n * sizeof(T)) == cudaSuccess) {
      return pointer;
    }
    throw std::bad_alloc();
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    cudaFree(pointer);
  }
};

template <class T, class U>
__host__ __device__ bool operator==(const ManagedAllocator<T>&, const ManagedAllocator<U>&) {
  return true;
}

template <class T, class U>
__host__ __device__ bool operator==(const ManagedAllocator<T>&, const DeviceAllocator<U>&) {
  return true;
}

template <class T, class U>
__host__ __device__ bool operator!=(const ManagedAllocator<T>&, const ManagedAllocator<U>&) {
  return false;
}

template <class T, class U>
__host__ __device__ bool operator!=(const ManagedAllocator<T>&, const DeviceAllocator<U>&) {
  return false;
}


}

#endif //CUDAPP_MANAGED_ALLOCATOR_H

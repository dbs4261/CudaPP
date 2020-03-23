//
// Created by Daniel Simon on 8/12/19.
//

#ifndef CUDAPP_CUDA_ARRAY_H
#define CUDAPP_CUDA_ARRAY_H

#include "include/cudapp/utilities/ide_helpers.h"

#include <cassert>
#include <utility>

#include <channel_descriptor.h>

#include "include/cudapp/exceptions/cuda_exception.h"
#include "include/cudapp/utilities/memory_helpers.h"
#include "include/cudapp/mathematics/vector_type_traits.h"

namespace cudapp {

template <typename T>
class CudaArray {
  static_assert(not std::is_same<T, void>::value, "Template type cannot be void");
 public:
  CudaArray() noexcept(false) : CudaArray<T>(1, 1, 1) {}
  CudaArray(std::size_t w, std::size_t h=0, std::size_t d=0, unsigned int _flags=0) noexcept(false)
      : CudaArray(make_cudaExtent(w, h, d), _flags) {}
  explicit CudaArray(cudaExtent _extent, unsigned int _flags=0) noexcept(false)
      : extent(_extent), array(nullptr), flags(_flags) {
    auto desc = cudaCreateChannelDesc<T>();
    cudaError_t ret = cudaMalloc3DArray(&array, &desc, extent, flags);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }
  CudaArray(const T* _data, std::size_t w, std::size_t h=0, std::size_t d=0, unsigned int _flags=0) noexcept(false)
      : CudaArray<T>(_data, make_cudaExtent(w, h, d), _flags) {}
  CudaArray(const T* _data, cudaExtent _extent, unsigned int _flags=0) noexcept(false)
      : CudaArray<T>(_extent, _flags) {
    this->Set(_data);
  }

  CudaArray(const CudaArray<T>& other) noexcept(false) : CudaArray(other, other.flags) {}
  CudaArray(const CudaArray<T>& other, unsigned int _flags=0) noexcept(false) : CudaArray(other.extent, _flags) {
    this->Set(other);
  }
  CudaArray(CudaArray<T>&& other) noexcept : extent(std::move(extent)), array(std::move(other.array)), flags(other.flags) {
    other.array = nullptr;
  }

  ~CudaArray() noexcept {
    cudaError_t ret = cudaFreeArray(this->array);
    if (ret != cudaSuccess) {
      // Uncatchable warning
      throw CudaException(ret);
    }
  }

  CudaArray<T>& operator=(const CudaArray<T>& other) noexcept(false) {
    if (other.array != this->array) {
      cudaError_t ret = cudaFreeArray(this->array);
      if (ret != cudaSuccess) {
        throw CudaException(ret);
      }
      this->extent = other.extent;
      this->flags = other.flags;
      this->array = other.array;
    }
    return *this;
  }

  CudaArray<T>& operator=(CudaArray<T>&& other) noexcept {
    this->extent = other.extent;
    this->flags = other.flags;
    // We swap instead of moving the pointer. That way this->array gets destroyed when other is destructed.
    std::swap(this->array, other.array);
    return *this;
  }

  const cudaExtent& Extent() const noexcept {
    return this->extent;
  }

  cudaArray* ArrayPtr() noexcept{
    return this->array;
  }
  const cudaArray* ArrayPtr() const noexcept {
    return this->array;
  }

  void Set(const T* _data) noexcept(false) {
    cudaMemcpy3DParms params = Memcpy3DParamsHD<T>(_data, this->array, this->extent);
    cudaError_t ret = cudaMemcpy3D(&params);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  void Set(const CudaArray<T>& other) noexcept(false) {
    if (this->array != other.array) {
      assert(this->Size() == other.Size());
      cudaMemcpy3DParms params = Memcpy3DParamsDD(other.array, this->array, this->extent);
      cudaError_t ret = cudaMemcpy3D(&params);
      if (ret != cudaSuccess) {
        throw CudaException(ret);
      }
    }
  }

  void Get(T* _data) const noexcept(false) {
    cudaMemcpy3DParms params = Memcpy3DParamsDH<T>(this->array, _data, this->extent);
    cudaError_t ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    ret = cudaMemcpy3D(&params);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  std::size_t Size() const noexcept {
    return (this->extent.width != 0 ? this->extent.width : 1) *
           (this->extent.height != 0 ? this->extent.height : 1) *
           (this->extent.depth != 0 ? this->extent.depth : 1);
  }

  std::size_t Width() const noexcept {
    return this->extent.width;
  }

  std::size_t Height() const noexcept {
    return this->extent.height;
  }

  std::size_t Depth() const noexcept {
    return this->extent.depth;
  }

 protected:
  cudaExtent extent;
  cudaArray* array;
  unsigned int flags;
};

}

#endif //CUDAPP_CUDA_ARRAY_H

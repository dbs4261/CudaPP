//
// Created by Daniel Simon on 8/12/19.
//

#ifndef CUDAPP_CUDA_ARRAY_H
#define CUDAPP_CUDA_ARRAY_H

#include "utilities/ide_helpers.h"

#include <cassert>
#include <utility>

#include <channel_descriptor.h>

#include "utilities/memory_helpers.h"
#include "vector_types/vector_type_traits.h"

namespace cudapp {

template <typename T>
class CudaArray {
  static_assert(not std::is_same<T, void>::value, "Template type cannot be void");
 public:
  CudaArray() : CudaArray<T>(1, 1, 1) {}
  CudaArray(std::size_t w, std::size_t h=0, std::size_t d=0, unsigned int _flags=0)
      : CudaArray(make_cudaExtent(w, h, d), _flags) {}
  explicit CudaArray(cudaExtent _extent, unsigned int _flags=0)
      : extent(_extent), array(nullptr), flags(_flags) {
    auto desc = cudaCreateChannelDesc<T>();
    CudaCatchError(cudaMalloc3DArray(&array, &desc, extent, flags));
  }
  CudaArray(const T* _data, std::size_t w, std::size_t h=0, std::size_t d=0, unsigned int _flags=0)
      : CudaArray<T>(_data, make_cudaExtent(w, h, d), _flags) {}
  CudaArray(const T* _data, cudaExtent _extent, unsigned int _flags=0)
      : CudaArray<T>(_extent, _flags) {
    this->Set(_data);
  }

  CudaArray(const CudaArray<T>& other) : CudaArray(other, other.flags) {}
  CudaArray(const CudaArray<T>& other, unsigned int _flags=0) : CudaArray(other.extent, _flags) {
    this->Set(other);
  }
  CudaArray(CudaArray<T>&& other) noexcept : extent(std::move(extent)), array(std::move(other.array)), flags(other.flags) {
    other.array = nullptr;
  }

  ~CudaArray() {
    CudaCatchError(cudaFreeArray(this->array));
  }

  CudaArray<T>& operator=(const CudaArray<T>& other) {
    if (other.array != this->array) {
      CudaCatchError(cudaFreeArray(this->array));
    }
    this->extent = other.extent;
    this->flags = other.flags;
    this->array = other.array;
    return *this;
  }

  CudaArray<T>& operator=(CudaArray<T>&& other) noexcept {
    this->extent = other.extent;
    this->flags = other.flags;
    // We swap instead of moving the pointer. That way this->array gets destroyed when other is destructed.
    std::swap(this->array, other.array);
    return *this;
  }

  const cudaExtent& Extent() const {
    return this->extent;
  }

  cudaArray* ArrayPtr() {
    return this->array;
  }
  const cudaArray* ArrayPtr() const {
    return this->array;
  }

  void Set(const T* _data) {
    cudaMemcpy3DParms params = Memcpy3DParamsHD<T>(_data, this->array, this->extent);
    CudaCatchError(cudaMemcpy3D(&params));
  }

  void Set(const CudaArray<T>& other) {
    assert(this->Size() == other.Size());
    cudaMemcpy3DParms params = Memcpy3DParamsDD(other.array, this->array, this->extent);
    CudaCatchError(cudaMemcpy3D(&params));
  }

  void Get(T* _data) const {
    cudaMemcpy3DParms params = Memcpy3DParamsDH<T>(this->array, _data, this->extent);
    CudaCatchError(cudaDeviceSynchronize());
    CudaCatchError(cudaMemcpy3D(&params));
  }

  std::size_t Size() const {
    return (this->extent.width != 0 ? this->extent.width : 1) *
           (this->extent.height != 0 ? this->extent.height : 1) *
           (this->extent.depth != 0 ? this->extent.depth : 1);
  }

  std::size_t Width() const {
    return this->extent.width;
  }

  std::size_t Height() const {
    return this->extent.height;
  }

  std::size_t Depth() const {
    return this->extent.depth;
  }

 protected:
  cudaExtent extent;
  cudaArray* array;
  unsigned int flags;
};

}

#endif //CUDAPP_CUDA_ARRAY_H

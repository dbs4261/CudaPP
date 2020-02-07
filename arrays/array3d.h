//
// Created by Daniel Simon on 8/12/19.
//

#ifndef CUDAPP_ARRAY3D_H
#define CUDAPP_ARRAY3D_H

#include "utilities/ide_helpers.h"

#include <cassert>
#include <utility>

#include <channel_descriptor.h>

#include "utilities/memory_helpers.h"
#include "vector_types/vector_type_traits.h"

namespace cuda {

template <typename T>
class Array3D {
  static_assert(not std::is_same<T, void>::value, "Template type cannot be void");
 public:
  Array3D() : Array3D<T>(1, 1, 1) {}
  Array3D(std::size_t w, std::size_t h=0, std::size_t d=0, unsigned int _flags=0)
      : Array3D(make_cudaExtent(w, h, d), _flags) {}
  explicit Array3D(cudaExtent _extent, unsigned int _flags=0)
      : extent(_extent), array(nullptr), flags(_flags) {
    auto desc = cudaCreateChannelDesc<T>();
    CudaCatchError(cudaMalloc3DArray(&array, &desc, extent, flags));
  }
  Array3D(const T* _data, std::size_t w, std::size_t h=0, std::size_t d=0, unsigned int _flags=0)
      : Array3D<T>(_data, make_cudaExtent(w, h, d), _flags) {}
  Array3D(const T* _data, cudaExtent _extent, unsigned int _flags=0)
      : Array3D<T>(_extent, _flags) {
    this->Set(_data);
  }

  Array3D(const Array3D<T>& other) : Array3D(other, other.flags) {}
  Array3D(const Array3D<T>& other, unsigned int _flags=0) : Array3D(other.extent, _flags) {
    this->Set(other);
  }
  Array3D(Array3D<T>&& other) noexcept : extent(std::move(extent)), array(std::move(other.array)), flags(other.flags) {
    other.array = nullptr;
  }

  ~Array3D() {
    CudaCatchError(cudaFreeArray(this->array));
  }

  Array3D<T>& operator=(const Array3D<T>& other) {
    if (other.array != this->array) {
      CudaCatchError(cudaFreeArray(this->array));
    }
    this->extent = other.extent;
    this->flags = other.flags;
    this->array = other.array;
    return *this;
  }

  Array3D<T>& operator=(Array3D<T>&& other) noexcept {
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

  void Set(const Array3D<T>& other) {
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

#endif //CUDAPP_ARRAY3D_H

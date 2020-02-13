//
// Created by Daniel Simon on 8/12/19.
//

#ifndef CUDAPP_MIPMAPPED_ARRAY_H
#define CUDAPP_MIPMAPPED_ARRAY_H

#include "utilities/ide_helpers.h"

#include <cassert>
#include <utility>

#include <channel_descriptor.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda.h>

#include "utilities/memory_helpers.h"
#include "vector_types/vector_type_traits.h"

#include "cuda_array.h"

namespace cuda {

template <typename T>
struct MipmappedArray {
 public:
  MipmappedArray() : MipmappedArray<T>(1, 1, 1, 1) {}
  MipmappedArray(std::size_t w, std::size_t h, std::size_t d, std::size_t l, unsigned int _flags=0)
      : MipmappedArray<T>(make_cudaExtent(w, h, d), l, _flags) {}
  MipmappedArray(cudaExtent _extent, std::size_t _levels, unsigned int _flags=0)
      : extent(_extent), levels(_levels), array(nullptr), flags(_flags) {
    auto desc = cudaCreateChannelDesc<T>();
    CudaCatchError(cudaMallocMipmappedArray(&array, &desc, extent, levels, flags));
  }
  MipmappedArray(const T* _data, std::size_t w, std::size_t h, std::size_t d, std::size_t l, unsigned int _flags=0)
      : MipmappedArray<T>(_data, make_cudaExtent(w, h, d), l, _flags) {}
  MipmappedArray(const T* _data, cudaExtent _extent, std::size_t _levels, unsigned int _flags=0)
      : MipmappedArray<T>(_extent, _levels, _flags) {
    this->Set(_data);
  }

  MipmappedArray(const MipmappedArray<T>& other) : MipmappedArray<T>(other, other.flags) {}
  MipmappedArray(const MipmappedArray<T>& other, unsigned int _flags=0) : MipmappedArray<T>(other.extent, other.levels, _flags) {
    this->Set(other);
  }
  MipmappedArray(MipmappedArray<T>&& other) noexcept : extent(std::move(other.extent)),
      levels(other.levels), array(std::move(other.array)), flags(other.flags) {
    other.array = nullptr;
  }

  ~MipmappedArray() {
    CudaCatchError(cudaFreeMipmappedArray(array));
  }
  
  MipmappedArray<T>& operator=(const MipmappedArray<T>& other) {
    if (other.array != array) {
      CudaCatchError(cudaFreeMipmappedArray(array));
    }
    extent = other.extent;
    levels = other.levels;
    array = other.array;
    flags = other.flags;
    return *this;
  }

  cudaMipmappedArray* MipmappedArrayPtr() {
    return array;
  }
  const cudaMipmappedArray* MipmappedArrayPtr() const {
    return array;
  }

  cudaArray* ArrayPtr(unsigned int level) {
    assert(level < levels);
    cudaArray* out;
    CudaCatchError(cudaGetMipmappedArrayLevel(&out, array, level));
    return out;
  }

  const cudaArray* ArrayPtr(unsigned int level) const {
    assert(level < levels);
    cudaArray* out;
    CudaCatchError(cudaGetMipmappedArrayLevel(&out, array, level));
    return out;
  }

  // NOTE: if the array will be writen to with a surface, then it requires the cudaArraySurfaceLoadStore flag.
  void Set(const T* _data, size_t l=0) {
    cudaExtent temp;
    CudaCatchError(cudaArrayGetInfo(nullptr, &temp, nullptr, this->ArrayPtr(l)));
    cudaMemcpy3DParms params = Memcpy3DParamsHD<T>(_data, this->ArrayPtr(l), temp);
    CudaCatchError(cudaMemcpy3D(&params));
    this->Pyramid();
  }

  void Set(const CudaArray<T>& other) {
    assert(this->Size() == other.Size());
    cudaMemcpy3DParms params = Memcpy3DParamsDD(other.array, this->ArrayPtr(0), extent);
    CudaCatchError(cudaMemcpy3D(&params));
    this->Pyramid();
  }

  void Set(const MipmappedArray<T>& other) {
    assert(this->Size() == other.Size());
    assert(this->levels == other.levels);
    cudaMemcpy3DParms params = BlankMemcpy3DParams();
    params.kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    for (std::size_t l = 0; l < this->levels; l++) {
      params.srcArray = const_cast<cudaArray*>(other.ArrayPtr(l));
      params.dstArray = this->ArrayPtr(l);
      CudaCatchError(cudaArrayGetInfo(nullptr, &params.extent, nullptr, params.srcArray));
      CudaCatchError(cudaMemcpy3D(&params));
    }
  }

  std::size_t Size() const {
    return (extent.width != 0 ? extent.width : 1) *
           (extent.height != 0 ? extent.height : 1) *
           (extent.depth != 0 ? extent.depth : 1);
  }

  std::size_t Width() const {
    return extent.width;
  }

  std::size_t Height() const {
    return extent.height;
  }

  std::size_t Depth() const {
    return extent.depth;
  }
  
  std::size_t Levels() const {
    return levels;
  }

 protected:
  void Pyramid() {
    cudaExtent level_extent;
    for (std::size_t l = 0; l < this->levels; l++) {
      CudaCatchError(cudaArrayGetInfo(nullptr, &level_extent, nullptr, this->ArrayPtr(l)));
    }
  }

  cudaExtent extent;
  std::size_t levels;
  cudaMipmappedArray* array;
  unsigned int flags;
};

}

#endif //CUDAPP_MIPMAPPED_ARRAY_H

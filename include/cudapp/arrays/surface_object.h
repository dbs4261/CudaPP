//
// Created by Daniel Simon on 8/14/19.
//

#ifndef CUDAPP_SURFACE_OBJECT_H
#define CUDAPP_SURFACE_OBJECT_H

#include <cassert>
#include <memory>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <surface_types.h>
#include <surface_indirect_functions.h>

#include "cudapp/utilities/macros.h"
#include "cudapp/utilities/memory_helpers.h"

#include "cuda_array.h"
#include "surface_view.h"

namespace cudapp {

template <typename T, std::size_t Dimensions, bool NormalizedFloat, bool NormalizedCoordinates>
class TextureObject;

namespace detail {

template <typename T, std::size_t Dimensions, bool NormalizedFloat, bool NormalizedCoordinates>
class TextureObject;

template <typename T, std::size_t Dimensions>
class SurfaceObject {
 public:
  using value_type = T;
  static constexpr std::size_t Dims = Dimensions;

  explicit SurfaceObject(std::shared_ptr<CudaArray<T>> _array_ptr) : array_ptr(_array_ptr), surface(0) {}
  SurfaceObject(std::size_t w, std::size_t h, std::size_t d, unsigned int flags=0)
      : SurfaceObject(std::make_shared<CudaArray<T>>(w, h, d,
          flags | static_cast<unsigned int>(cudaArraySurfaceLoadStore))) {}
  SurfaceObject(cudaExtent _extent, unsigned int flags=0)
      : SurfaceObject(std::make_shared<CudaArray<T>>(_extent,
          flags | static_cast<unsigned int>(cudaArraySurfaceLoadStore))) {}
  SurfaceObject(const T* _data, std::size_t w, std::size_t h, std::size_t d, unsigned int flags=0)
      : SurfaceObject(std::make_shared<CudaArray<T>>(_data, w, h, d,
          flags | static_cast<unsigned int>(cudaArraySurfaceLoadStore))) {}
  SurfaceObject(const T* _data, cudaExtent _extent, unsigned int flags=0)
      : SurfaceObject(std::make_shared<CudaArray<T>>(_data, _extent,
          flags | static_cast<unsigned int>(cudaArraySurfaceLoadStore))) {}

  SurfaceObject(const SurfaceObject<T, Dims>& other) : array_ptr(std::make_shared<CudaArray<T>>(*other.array_ptr)), surface(0) {}
  SurfaceObject(SurfaceObject<T, Dims>&& other) : array_ptr(other.array_ptr), surface(0) {}

  ~SurfaceObject() {
    if (surface != 0) {
      CudaCatchError(cudaDestroySurfaceObject(surface));
    }
  }

  SurfaceObject& operator=(const SurfaceObject<T, Dims>& other) {
    if (surface != 0) {
      CudaCatchError(cudaDestroySurfaceObject(surface));
      surface = 0;
    }
    this->array_ptr = std::make_shared<CudaArray<T>>(*other.array_ptr);
    return *this;
  }

  SurfaceObject& operator=(SurfaceObject<T, Dims>&& other) {
    if (surface != 0) {
      CudaCatchError(cudaDestroySurfaceObject(surface));
      surface = 0;
    }
    this->array_ptr = other.array_ptr;
    return *this;
  }

  std::shared_ptr<CudaArray<T>> Array() {
    return this->array;
  }

  std::shared_ptr<const CudaArray<T>> Array() const {
    return this->array;
  }

  cudaResourceDesc ResourceDescription() const {
    cudaResourceDesc resource_desc = BlankResourceDesc();
    resource_desc.resType = cudaResourceType::cudaResourceTypeArray;
    resource_desc.res.array.array = array_ptr->ArrayPtr();
    return resource_desc;
  }

  cudapp::SurfaceView<T, Dimensions> View() {
    if (surface == 0) {
      cudaResourceDesc desc = this->ResourceDescription();
      CudaCatchError(cudaCreateSurfaceObject(&this->surface, &desc));
    }
    return cudapp::SurfaceView<T, Dimensions>(surface);
  }

  cudapp::SurfaceView<std::add_const_t<T>, Dimensions> View() const {
    if (surface == 0) {
      cudaResourceDesc desc = this->ResourceDescription();
      CudaCatchError(cudaCreateSurfaceObject(&this->surface, &desc));
    }
    return cudapp::SurfaceView<std::add_const_t<T>, Dimensions>(surface);
  }

  inline auto CView() const {
    return View();
  }

  void Set(const T* _data) {
    this->array_ptr->Set(_data);
  }

  void Set(const CudaArray<T>& other) {
    this->array_ptr->Set(other);
  }

  void Set(const SurfaceObject<T, Dimensions>& other) {
    this->array_ptr->Set(other.array_ptr);
  }

  void Get(T* _data) const {
    this->array_ptr->Get(_data);
  }

  friend cudapp::TextureObject<T, Dimensions, true, true>;
  friend cudapp::TextureObject<T, Dimensions, true, false>;
  friend cudapp::TextureObject<T, Dimensions, false, true>;
  friend cudapp::TextureObject<T, Dimensions, false, false>;

 protected:
  std::shared_ptr<CudaArray<T>> array_ptr;
  mutable cudaSurfaceObject_t surface;
};

}

template <typename T, std::size_t Dimensions, bool NormalizedFloat, bool NormalizedCoordinates>
class TextureObject;

template <typename T, std::size_t Dimensions>
class SurfaceObject : public detail::SurfaceObject<T, Dimensions> {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Surface must have 1, 2 or 3 dimensions");
};

template <typename T>
class SurfaceObject<T, 1> : public detail::SurfaceObject<T, 1> {
 public:
  using value_type = typename detail::SurfaceObject<T, 1>::value_type;
  static constexpr std::size_t Dims = detail::SurfaceObject<T, 1>::Dims;

  SurfaceObject() : SurfaceObject(1) {}
  explicit SurfaceObject(std::shared_ptr<CudaArray<T>> _array_ptr)
      : detail::SurfaceObject<T, Dims>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height == 0);
    assert(this->array_ptr->Extent().depth == 0);
  }
  SurfaceObject(std::size_t width) : detail::SurfaceObject<T, Dims>(width, 0, 0) {}
  SurfaceObject(const T* _data, std::size_t width) : detail::SurfaceObject<T, Dims>(_data, width) {}

  explicit SurfaceObject(const SurfaceObject<T, Dims>& other) : SurfaceObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  explicit SurfaceObject(SurfaceObject<T, Dims>&& other) : SurfaceObject(other.array_ptr) {}

  template <bool NormalizedFloat, bool NormalizedCoordinates>
  explicit SurfaceObject(const TextureObject<T, Dims, NormalizedFloat, NormalizedCoordinates>& other) : SurfaceObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  template <bool NormalizedFloat, bool NormalizedCoordinates>
  explicit SurfaceObject(TextureObject<T, Dims, NormalizedFloat, NormalizedCoordinates>&& other) : SurfaceObject(other.array_ptr) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  friend TextureObject<T, Dims, true, true>;
  friend TextureObject<T, Dims, true, false>;
  friend TextureObject<T, Dims, false, true>;
  friend TextureObject<T, Dims, false, false>;
};

template <typename T>
 class SurfaceObject<T, 2> : public detail::SurfaceObject<T, 2> {
 public:
  using value_type = typename detail::SurfaceObject<T, 2>::value_type;
  static constexpr std::size_t Dims = detail::SurfaceObject<T, 2>::Dims;

  SurfaceObject() : SurfaceObject(1, 1) {}
  explicit SurfaceObject(std::shared_ptr<CudaArray<T>> _array_ptr)
      : detail::SurfaceObject<T, Dims>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height != 0);
    assert(this->array_ptr->Extent().depth == 0);
  }
  SurfaceObject(std::size_t width, std::size_t height) : detail::SurfaceObject<T, Dims>(width, height, 0) {}
  SurfaceObject(const T* _data, std::size_t width, std::size_t height) : detail::SurfaceObject<T, Dims>(_data, width, height, 0) {}

  explicit SurfaceObject(const SurfaceObject<T, Dims>& other) : SurfaceObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  explicit SurfaceObject(SurfaceObject<T, Dims>&& other) : SurfaceObject(other.array_ptr) {}

  template <bool NormalizedFloat, bool NormalizedCoordinates>
  explicit SurfaceObject(const TextureObject<T, Dims, NormalizedFloat,NormalizedCoordinates>& other) : SurfaceObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  template <bool NormalizedFloat, bool NormalizedCoordinates>
  explicit SurfaceObject(TextureObject<T, Dims, NormalizedFloat, NormalizedCoordinates>&& other) : SurfaceObject(other.array_ptr) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Height() const {
    return this->array_ptr->Height();
  }

  friend TextureObject<T, Dims, true, true>;
  friend TextureObject<T, Dims, true, false>;
  friend TextureObject<T, Dims, false, true>;
  friend TextureObject<T, Dims, false, false>;
};

template <typename T>
class SurfaceObject<T, 3> : public detail::SurfaceObject<T, 3> {
 public:
  using value_type = typename detail::SurfaceObject<T, 3>::value_type;
  static constexpr std::size_t Dims = detail::SurfaceObject<T, 3>::Dims;

  SurfaceObject() : SurfaceObject(1, 1, 1) {}
  explicit SurfaceObject<T, 3>(std::shared_ptr<CudaArray<T>> _array_ptr)
      : detail::SurfaceObject<T, Dims>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height != 0);
    assert(this->array_ptr->Extent().depth != 0);
  }
  SurfaceObject(std::size_t width, std::size_t height, std::size_t depth) : detail::SurfaceObject<T, Dims>(width, height, depth) {}
  SurfaceObject(const T* _data, std::size_t width, std::size_t height, std::size_t depth) : detail::SurfaceObject<T, Dims>(_data, width, height, depth) {}

  explicit SurfaceObject(const SurfaceObject<T, Dims>& other) : SurfaceObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  explicit SurfaceObject(SurfaceObject<T, Dims>&& other) : SurfaceObject(other.array_ptr) {}

  template <bool NormalizedFloat, bool NormalizedCoordinates>
  explicit SurfaceObject(const TextureObject<T, Dims, NormalizedFloat, NormalizedCoordinates>& other) : SurfaceObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  template <bool NormalizedFloat, bool NormalizedCoordinates>
  explicit SurfaceObject(TextureObject<T, Dims, NormalizedFloat, NormalizedCoordinates>&& other) : SurfaceObject(other.array_ptr) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Height() const {
    return this->array_ptr->Height();
  }

  std::size_t Depth() const {
    return this->array_ptr->Depth();
  }

  friend TextureObject<T, Dims, true, true>;
  friend TextureObject<T, Dims, true, false>;
  friend TextureObject<T, Dims, false, true>;
  friend TextureObject<T, Dims, false, false>;
};

}

#endif //CUDAPP_SURFACE_OBJECT_H

//
// Created by Daniel Simon on 8/12/19.
//

#ifndef CUDAPP_TEXTURE_OBJECT_H
#define CUDAPP_TEXTURE_OBJECT_H

#include "cudapp/utilities/ide_helpers.h"

#include <memory>
#include <type_traits>

#include <driver_types.h>
#include <texture_indirect_functions.h>

#include "cudapp/mathematics/vector_type_traits.h"

#include "cuda_array.h"
#include "texture_view.h"
#include "surface_object.h"

namespace cudapp {

namespace detail {

template <typename T, std::size_t Dimensions>
class SurfaceObject;

namespace detail {

template <typename T, std::size_t Dimensions, bool NormalizedFloat, bool NormalizedCoordinates>
class TextureObject {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Texture must have 1, 2 or 3 dimensions");
 public:
  TextureObject(std::shared_ptr<CudaArray<T>> _array_ptr, cudaTextureDesc _texture_description)
      : texture_desc(_texture_description), array_ptr(_array_ptr), texture(0), modified(true) {
  #ifdef __cpp_if_constexpr
    if constexpr (NormalizedFloat) {
  #else
    if (NormalizedFloat) {
  #endif
      this->texture_desc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
    } else {
      this->texture_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
    }
    texture_desc.normalizedCoords = static_cast<int>(NormalizedCoordinates);
  }

  explicit TextureObject(std::shared_ptr<CudaArray<T>> _array_ptr)
      : TextureObject(_array_ptr, BlankTextureDesc()) {}

  TextureObject(std::size_t w, std::size_t h, std::size_t d, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : TextureObject(std::make_shared<CudaArray<T>>(w, h, d, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  TextureObject(cudaExtent _extent, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : TextureObject(std::make_shared<CudaArray<T>>(_extent, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  TextureObject(const T* _data, std::size_t w, std::size_t h, std::size_t d, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : TextureObject(std::make_shared<CudaArray<T>>(_data, w, h, d, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  TextureObject(const T* _data, cudaExtent _extent, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : TextureObject(std::make_shared<CudaArray<T>>(_data, _extent, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  // Both converters and copy/move
  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(const TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>& other)
      : TextureObject(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>&& other)
      : TextureObject(other.array_ptr, other.texture_desc) {}

  ~TextureObject() {
    if (this->texture != 0) {
      CudaCatchError(cudaDestroyTextureObject(this->texture));
    }
  }

  TextureObject& operator=(const TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>& other) {
    array_ptr = other.array_ptr;
    texture_desc = other.texture_desc;
    texture = other.texture;
    modified = other.modified;
    return *this;
  }

  TextureObject& operator=(TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>&& other) {
    array_ptr = other.array_ptr;
    texture_desc = other.texture_desc;
    texture = other.texture;
    other.texture = 0;
    modified = other.modified;
    return *this;
  }

  std::shared_ptr<CudaArray<T>> Array() {
    return this->array;
  }

  std::shared_ptr<const CudaArray<T>> Array() const {
    return this->array;
  }

  void SetAddressMode(cudaTextureAddressMode mode, int dimension) {
    if (dimension == 0 or dimension < 0) {
      texture_desc.addressMode[0] = mode;
    }
    if (dimension == 1 or dimension < 0) {
      texture_desc.addressMode[1] = mode;
    }
    if (dimension == 2 or dimension < 0) {
      texture_desc.addressMode[2] = mode;
    }
    modified = true;
  }

  void SetBilinearInterpolate(typename std::enable_if_t<NormalizedFloat or std::is_floating_point<T>::value, bool>::type interpolate) {
    if (interpolate) {
      texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    } else {
      texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
    }
    modified = true;
  }

  void SetSRGBCorrection(bool correct) {
    texture_desc.sRGB = static_cast<int>(correct);
    modified = true;
  }

  void SetBorderColor(float c0, float c1 = 0.0f, float c2 = 0.0f, float c3 = 0.0f) {
    texture_desc.borderColor[0] = c0;
    texture_desc.borderColor[1] = c1;
    texture_desc.borderColor[2] = c2;
    texture_desc.borderColor[3] = c3;
    modified = true;
  }

  void SetBorderColor(float4 val) {
    texture_desc.borderColor[0] = val.x;
    texture_desc.borderColor[1] = val.y;
    texture_desc.borderColor[2] = val.z;
    texture_desc.borderColor[3] = val.w;
    modified = true;
  }

  cudaResourceDesc ResourceDescription() const {
    cudaResourceDesc resource_desc;
    resource_desc.resType = cudaResourceType::cudaResourceTypeArray;
    resource_desc.res.array.array = array_ptr->ArrayPtr();
    return resource_desc;
  }

  cudaTextureDesc TextureDescription() const {
    return texture_desc;
  }

  void Set(T* _data) {
    this->array_ptr->Set(_data);
  }

  void Set(const CudaArray<T>& other) {
    this->array_ptr->Set(other);
  }

  void Set(const SurfaceObject<T, Dimensions>& other) {
    this->array_ptr->Set(other.array_ptr);
  }

  friend SurfaceObject<T, Dimensions>;

 protected:
  void UpdateTexture() const {
    if (modified) {
//      printf("Updating textures\n");
      cudaResourceDesc resource_desc = ResourceDescription();
      CudaCatchError(cudaCreateTextureObject(&this->texture, &resource_desc, &this->texture_desc, nullptr));
      modified = false;
    }
  }

  cudaTextureDesc texture_desc;
  std::shared_ptr<CudaArray<T>> array_ptr;
  mutable cudaTextureObject_t texture;
  mutable bool modified;
};

}

template <typename T, std::size_t Dimensions, bool _NormalizedFloat, bool _NormalizedCoordinates>
 class TextureObject : public cuda::detail::detail::TextureObject<T, Dimensions, _NormalizedFloat, _NormalizedCoordinates> {
 public:
  static constexpr bool NormalizedFloat = _NormalizedFloat;
  static constexpr bool NormalizedCoordinates = _NormalizedCoordinates;
  static constexpr int channels = vector_channels_v<T>;
  using value_type = composed_vector_type_t<float, channels>;
  using cuda::detail::detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>::TextureObject;

  cuda::TextureView<T, Dimensions, NormalizedCoordinates> View() const {
    this->UpdateTexture();
    return cuda::TextureView<T, Dimensions, NormalizedCoordinates>(this->texture);
  }
};

template <typename T, std::size_t Dimensions>
class TextureObject<T, Dimensions, false, true> : public cuda::detail::detail::TextureObject<T, Dimensions, false, true> {
 public:
  static constexpr bool NormalizedFloat = false;
  static constexpr bool NormalizedCoordinates = true;
  static constexpr int channels = vector_channels_v<T>;
  using value_type = composed_vector_type_t<float, channels>;
  using cuda::detail::detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>::TextureObject;

  cuda::TextureView<T, Dimensions, NormalizedCoordinates> View() const {
    this->UpdateTexture();
    return cuda::TextureView<T, Dimensions, NormalizedCoordinates>(this->texture,
        float3{this->array_ptr->Width(), this->array_ptr->Height(), this->array_ptr->Depth()});
  }
};

template <typename T, std::size_t Dimensions>
class TextureObject<T, Dimensions, true, false> : public cuda::detail::detail::TextureObject<T, Dimensions, true, false> {
 public:
  static constexpr bool NormalizedFloat = true;
  static constexpr bool NormalizedCoordinates = false;
  static constexpr int channels = vector_channels_v<T>;
  using value_type = composed_vector_type_t<float, channels>;
  using cuda::detail::detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>::TextureObject;

  cuda::TextureView<value_type, Dimensions, NormalizedCoordinates> View() const {
    this->UpdateTexture();
    return cuda::TextureView<value_type, Dimensions, NormalizedCoordinates>(this->texture);
  }
};

template <typename T, std::size_t Dimensions>
class TextureObject<T, Dimensions, true, true> : public cuda::detail::detail::TextureObject<T, Dimensions, true, true> {
 public:
  static constexpr bool NormalizedFloat = true;
  static constexpr bool NormalizedCoordinates = true;
  static constexpr int channels = vector_channels_v<T>;
  using value_type = composed_vector_type_t<float, channels>;
  using cuda::detail::detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>::TextureObject;

  cuda::TextureView<value_type, Dimensions, NormalizedCoordinates> View() const {
    this->UpdateTexture();
    return cuda::TextureView<value_type, Dimensions, NormalizedCoordinates>(this->texture,
        float3{this->array_ptr->Width(), this->array_ptr->Height(), this->array_ptr->Depth()});
  }
};

}

template <typename T, std::size_t Dimensions>
class SurfaceObject;

template <typename T, std::size_t Dimensions, bool NormalizedFloat, bool NormalizedCoordinates>
 class TextureObject : public detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates> {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Texture must have 1, 2 or 3 dimensions");
};

template <typename T, bool NormalizedFloat, bool NormalizedCoordinates>
class TextureObject<T, 1, NormalizedFloat, NormalizedCoordinates> : public detail::TextureObject<T, 1, NormalizedFloat, NormalizedCoordinates> {
 public:
  static constexpr int Dimensions = 1;
  TextureObject() : TextureObject(1) {}
  explicit TextureObject(std::shared_ptr<CudaArray<T>> _array_ptr)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height == 0);
    assert(this->array_ptr->Extent().depth == 0);
  }
  TextureObject(std::size_t width)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(width, 0, 0) {}
  TextureObject(const T* _data, std::size_t width)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(_data, width, 0, 0) {}

  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(const TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>& other)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>&& other)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(other.array_ptr, other.texture_desc) {}
  explicit TextureObject(const SurfaceObject<T, Dimensions>& other)
      : TextureObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  explicit TextureObject(SurfaceObject<T, Dimensions>&& other)
      : TextureObject(other.array_ptr) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  friend SurfaceObject<T, Dimensions>;
};

template <typename T, bool NormalizedFloat, bool NormalizedCoordinates>
 class TextureObject<T, 2, NormalizedFloat, NormalizedCoordinates> : public detail::TextureObject<T, 2, NormalizedFloat, NormalizedCoordinates> {
 public:
  static constexpr int Dimensions = 2;
  TextureObject() : TextureObject(1, 1) {}
  explicit TextureObject(std::shared_ptr<CudaArray<T>> _array_ptr)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height != 0);
    assert(this->array_ptr->Extent().depth == 0);
  }
  TextureObject(std::size_t width, std::size_t height)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(width, height, 0) {}
  TextureObject(const T* _data, std::size_t width, std::size_t height)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(_data, width, height, 0) {}

  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(const TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>& other)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>&& other)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(other.array_ptr, other.texture_desc) {}
  explicit TextureObject(const SurfaceObject<T, Dimensions>& other)
      : TextureObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  explicit TextureObject(SurfaceObject<T, Dimensions>&& other)
      : TextureObject(other.array_ptr) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Height() const {
    return this->array_ptr->Height();
  }

  friend SurfaceObject<T, Dimensions>;
};

template <typename T, bool NormalizedFloat, bool NormalizedCoordinates>
 class TextureObject<T, 3, NormalizedFloat, NormalizedCoordinates> : public detail::TextureObject<T, 3, NormalizedFloat, NormalizedCoordinates> {
 public:
  static constexpr int Dimensions = 3;
  TextureObject() : TextureObject(1, 1, 1) {}
  explicit TextureObject(std::shared_ptr<CudaArray<T>> _array_ptr)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height != 0);
    assert(this->array_ptr->Extent().depth != 0);
  }
  TextureObject(std::size_t width, std::size_t height, std::size_t depth)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(width, height, depth) {}
  TextureObject(const T* _data, std::size_t width, std::size_t height, std::size_t depth)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(_data, width, height, depth) {}

  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(const TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>& other)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat, bool OtherNormalizedCoordinates>
  explicit TextureObject(TextureObject<T, Dimensions, OtherNormalizedFloat, OtherNormalizedCoordinates>&& other)
      : detail::TextureObject<T, Dimensions, NormalizedFloat, NormalizedCoordinates>(other.array_ptr, other.texture_desc) {}
  explicit TextureObject(const SurfaceObject<T, Dimensions>& other)
      : TextureObject(std::make_shared<CudaArray<T>>(*other.array_ptr)) {}
  explicit TextureObject(SurfaceObject<T, Dimensions>&& other)
      : TextureObject(other.array_ptr) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Height() const {
    return this->array_ptr->Height();
  }

  std::size_t Depth() const {
    return this->array_ptr->Depth();
  }

  friend SurfaceObject<T, Dimensions>;
};

}

#endif //CUDAPP_TEXTURE_OBJECT_H

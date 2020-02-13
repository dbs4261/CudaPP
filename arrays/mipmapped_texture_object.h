//
// Created by Daniel Simon on 10/5/19.
//

#ifndef CUDAPP_MIPMAPPED_TEXTURE_OBJECT_H
#define CUDAPP_MIPMAPPED_TEXTURE_OBJECT_H

#include "utilities/ide_helpers.h"

#include <memory>
#include <type_traits>

#include <driver_types.h>
#include <texture_indirect_functions.h>

#include "vector_types/vector_type_traits.h"

#include "mipmapped_array.h"
#include "texture_view.h"
#include "surface_object.h"

namespace cudapp {

namespace detail {

template <typename T, std::size_t Dimensions>
class SurfaceObject;

namespace detail {

template <typename T, std::size_t Dimensions, bool NormalizedFloat>
class MipmappedTextureObject {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Mipmapped Texture must have 1, 2 or 3 dimensions");
  static_assert(std::is_floating_point<T>::value or NormalizedFloat, "The result of texture addressing must be a float or else there is no point in using a mipmap");
 public:
  static constexpr bool NormalizedCoordinates = true;
  MipmappedTextureObject(std::shared_ptr<MipmappedArray<T>> _array_ptr, cudaTextureDesc _texture_description)
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

  explicit MipmappedTextureObject(std::shared_ptr<MipmappedArray<T>> _array_ptr)
      : MipmappedTextureObject(_array_ptr, BlankTextureDesc()) {}

  MipmappedTextureObject(std::size_t w, std::size_t h, std::size_t d, std::size_t l, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : MipmappedTextureObject(std::make_shared<MipmappedArray<T>>(w, h, d, l, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  MipmappedTextureObject(cudaExtent _extent, std::size_t l, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : MipmappedTextureObject(std::make_shared<MipmappedArray<T>>(_extent, l, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  MipmappedTextureObject(const T* _data, std::size_t w, std::size_t h, std::size_t d, std::size_t l, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : MipmappedTextureObject(std::make_shared<MipmappedArray<T>>(_data, w, h, d, l, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  MipmappedTextureObject(const T* _data, cudaExtent _extent, std::size_t l, cudaTextureDesc _texture_description = BlankTextureDesc(), unsigned int flags = 0)
      : MipmappedTextureObject(std::make_shared<MipmappedArray<T>>(_data, _extent, l, flags | static_cast<unsigned int>(cudaArrayTextureGather)), _texture_description) {}

  // Both converters and copy/move
  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(const MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>& other)
      : MipmappedTextureObject(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>&& other)
      : MipmappedTextureObject(other.array_ptr, other.texture_desc) {}

  ~MipmappedTextureObject() {
    if (this->texture != 0) {
      CudaCatchError(cudaDestroyTextureObject(this->texture));
    }
  }

  MipmappedTextureObject& operator=(const MipmappedTextureObject<T, Dimensions, NormalizedCoordinates>& other) {
    array_ptr = other.array_ptr;
    texture_desc = other.texture_desc;
    texture = other.texture;
    modified = other.modified;
    return *this;
  }

  MipmappedTextureObject& operator=(MipmappedTextureObject<T, Dimensions, NormalizedCoordinates>&& other) {
    array_ptr = other.array_ptr;
    texture_desc = other.texture_desc;
    texture = other.texture;
    other.texture = 0;
    modified = other.modified;
    return *this;
  }

  std::shared_ptr<MipmappedArray<T>> Array() {
    return this->array_ptr;
  }

  std::shared_ptr<const MipmappedArray<T>> Array() const {
    return this->array_ptr;
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

  void SetBilinearInterpolate(typename std::enable_if<NormalizedFloat or std::is_floating_point<T>::value, bool>::type interpolate) {
    if (interpolate) {
      texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    } else {
      texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
    }
    modified = true;
  }

  void SetTrilinearInterpolate(typename std::enable_if<NormalizedFloat or std::is_floating_point<T>::value, bool>::type interpolate) {
    if (interpolate) {
      texture_desc.mipmapFilterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    } else {
      texture_desc.mipmapFilterMode = cudaTextureFilterMode::cudaFilterModePoint;
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

  void SetMaxAnisotropy(unsigned int val) {
    texture_desc.maxAnisotropy = val;
    modified = true;
  }

  void SetMipmapLevelClamp(float minimum, float maximum) {
    texture_desc.minMipmapLevelClamp = minimum;
    texture_desc.maxMipmapLevelClamp = maximum;
    modified = true;
  }

  cudaResourceDesc ResourceDescription() const {
    cudaResourceDesc resource_desc;
    resource_desc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    resource_desc.res.mipmap.mipmap = array_ptr->MipmappedArrayPtr();
    return resource_desc;
  }

  cudaTextureDesc TextureDescription() const {
    return texture_desc;
  }

  void Set(T* _data) {
    this->array_ptr->Set(_data);
  }

  void Set(const MipmappedArray<T>& other) {
    this->array_ptr->Set(other);
  }

  void Set(const SurfaceObject<T, Dimensions>& other) {
    this->array_ptr->Set(other.array_ptr);
  }

  friend SurfaceObject<T, Dimensions>;

 protected:
  void UpdateTexture() const {
    if (modified) {
      cudaResourceDesc resource_desc = ResourceDescription();
      CudaCatchError(cudaCreateTextureObject(&this->texture, &resource_desc, &this->texture_desc, nullptr));
      modified = false;
    }
  }

  cudaTextureDesc texture_desc;
  std::shared_ptr<MipmappedArray<T>> array_ptr;
  mutable cudaTextureObject_t texture;
  mutable bool modified;
};

}

template <typename T, std::size_t Dimensions, bool NormalizedFloat>
class MipmappedTextureObject : public cuda::detail::detail::MipmappedTextureObject<T, Dimensions, false> {
 public:
  static constexpr int channels = vector_channels_v<T>;
  using value_type = T;
  using cuda::detail::detail::MipmappedTextureObject<T, Dimensions, false>::MipmappedTextureObject;

  cuda::TextureView<value_type, Dimensions, true> View() const {
    this->UpdateTexture();
    return cuda::TextureView<value_type, Dimensions, true>(this->texture,
        float3{this->array_ptr->Width(), this->array_ptr->Height(), this->array_ptr->Depth()});
  }
};

template <typename T, std::size_t Dimensions>
class MipmappedTextureObject<T, Dimensions, true> : public cuda::detail::detail::MipmappedTextureObject<T, Dimensions, true> {
 public:
  static constexpr int channels = vector_channels_v<T>;
  using value_type = composed_vector_type_t<float, channels>;
  using cuda::detail::detail::MipmappedTextureObject<T, Dimensions, true>::MipmappedTextureObject;

  cuda::TextureView<value_type, Dimensions, true> View() const {
    this->UpdateTexture();
    return cuda::TextureView<value_type, Dimensions, true>(this->texture,
        float3{this->array_ptr->Width(), this->array_ptr->Height(), this->array_ptr->Depth()});
  }
};

}

template <typename T, std::size_t Dimensions>
class SurfaceObject;

template <typename T, std::size_t Dimensions, bool NormalizedFloat>
class MipmappedTextureObject : public detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat> {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Texture must have 1, 2 or 3 dimensions");
};

template <typename T, bool NormalizedFloat>
class MipmappedTextureObject<T, 1, NormalizedFloat> : public detail::MipmappedTextureObject<T, 1, NormalizedFloat> {
 public:
  static constexpr int Dimensions = 1;
  MipmappedTextureObject() : MipmappedTextureObject(1, 1) {}
  explicit MipmappedTextureObject(std::shared_ptr<MipmappedArray<T>> _array_ptr)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height == 0);
    assert(this->array_ptr->Extent().depth == 0);
  }
  MipmappedTextureObject(std::size_t width, std::size_t _levels)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(width, 0, 0, _levels) {}
  MipmappedTextureObject(const T* _data, std::size_t width, std::size_t _levels)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(_data, width, 0, 0, _levels) {}

  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(const MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>& other)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>&& other)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(other.array_ptr, other.texture_desc) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Levels() const {
    return this->array_ptr->Levels();
  }
};

template <typename T, bool NormalizedFloat>
class MipmappedTextureObject<T, 2, NormalizedFloat> : public detail::MipmappedTextureObject<T, 2, NormalizedFloat> {
 public:
  static constexpr int Dimensions = 2;
  MipmappedTextureObject() : MipmappedTextureObject(1, 1, 1) {}
  explicit MipmappedTextureObject(std::shared_ptr<MipmappedArray<T>> _array_ptr)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height != 0);
    assert(this->array_ptr->Extent().depth == 0);
  }
  MipmappedTextureObject(std::size_t width, std::size_t height, std::size_t _levels)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(width, height, 0, _levels) {}
  MipmappedTextureObject(const T* _data, std::size_t width, std::size_t height, std::size_t _levels)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(_data, width, height, 0, _levels) {}

  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(const MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>& other)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>&& other)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(other.array_ptr, other.texture_desc) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Height() const {
    return this->array_ptr->Height();
  }

  std::size_t Levels() const {
    return this->array_ptr->Levels();
  }
};

template <typename T, bool NormalizedFloat>
class MipmappedTextureObject<T, 3, NormalizedFloat> : public detail::MipmappedTextureObject<T, 3, NormalizedFloat> {
 public:
  static constexpr int Dimensions = 3;
  MipmappedTextureObject() : MipmappedTextureObject(1, 1, 1, 1) {}
  explicit MipmappedTextureObject(std::shared_ptr<MipmappedArray<T>> _array_ptr)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(_array_ptr) {
    assert(this->array_ptr->Extent().width != 0);
    assert(this->array_ptr->Extent().height != 0);
    assert(this->array_ptr->Extent().depth != 0);
  }
  MipmappedTextureObject(std::size_t width, std::size_t height, std::size_t depth, std::size_t _levels)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(width, height, depth, _levels) {}
  MipmappedTextureObject(const T* _data, std::size_t width, std::size_t height, std::size_t depth, std::size_t _levels)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(_data, width, height, depth, _levels) {}

  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(const MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>& other)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(other.array_ptr, other.texture_desc) {}
  template <bool OtherNormalizedFloat>
  explicit MipmappedTextureObject(MipmappedTextureObject<T, Dimensions, OtherNormalizedFloat>&& other)
      : detail::MipmappedTextureObject<T, Dimensions, NormalizedFloat>(other.array_ptr, other.texture_desc) {}

  std::size_t Width() const {
    return this->array_ptr->Width();
  }

  std::size_t Height() const {
    return this->array_ptr->Height();
  }

  std::size_t Depth() const {
    return this->array_ptr->Depth();
  }

  std::size_t Levels() const {
    return this->array_ptr->Levels();
  }
};

}

#endif //CUDAPP_MIPMAPPED_TEXTURE_OBJECT_H

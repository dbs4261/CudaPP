//
// Created by Daniel Simon on 10/4/19.
//

#ifndef CUDAPP_TEXTURE_VIEW_H
#define CUDAPP_TEXTURE_VIEW_H

#include "utilities/ide_helpers.h"

#include <utility>

#include <texture_types.h>
#include <texture_indirect_functions.h>

namespace cudapp {

namespace detail {

template <typename T, std::size_t Dimensions, bool NormalizeCoordinates>
class TextureView {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Texture must have 1, 2 or 3 dimensions");
 public:
  using value_type = T;
  static constexpr std::size_t Dims = Dimensions;

  __host__ __device__ TextureView() noexcept : tex(0){}
  __host__ __device__ explicit TextureView(cudaTextureObject_t _tex) noexcept : tex(_tex) {}
  template <bool OtherNormalizedCoordinates>
  __host__ __device__ explicit TextureView(const TextureView<T, Dims, OtherNormalizedCoordinates>& _tex) noexcept : tex(_tex.tex) {}
  template <bool OtherNormalizedCoordinates>
  __host__ __device__ explicit TextureView(TextureView<T, Dims, OtherNormalizedCoordinates>&& _tex) noexcept : tex(std::move(_tex.tex)) {}

  template <bool OtherNormalizedCoordinates>
  __host__ __device__ __forceinline__ TextureView& operator=(const TextureView<T, Dims, OtherNormalizedCoordinates>& other) noexcept {
    this->tex = other.tex;
  }
  template <bool OtherNormalizedCoordinates>
  __host__ __device__ __forceinline__ TextureView& operator=(TextureView<T, Dims, OtherNormalizedCoordinates>&& other) noexcept {
    this->tex = std::move(other.tex);
  }

 protected:
  cudaTextureObject_t tex;
};

template <typename T, std::size_t Dimensions>
class TextureView<T, Dimensions, true> {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Texture must have 1, 2 or 3 dimensions");
 public:
  using value_type = T;
  static constexpr std::size_t Dims = Dimensions;

  __host__ __device__ TextureView() noexcept : tex(0), base_extent(float3{0, 0, 0}) {}
  __host__ __device__ explicit TextureView(cudaTextureObject_t _tex, float3 _base_extent) noexcept : tex(_tex), base_extent(_base_extent) {}
  __host__ __device__ explicit TextureView(const TextureView<T, Dims, true>& _tex) noexcept : tex(_tex.tex), base_extent(_tex.base_extent) {}
  __host__ __device__ explicit TextureView(TextureView<T, Dims, true>&& _tex) noexcept : tex(std::move(_tex.tex)), base_extent(std::move(_tex.base_extent)) {}

  __host__ __device__ __forceinline__ TextureView& operator=(const TextureView<T, Dims, true>& other) noexcept {
    this->tex = other.tex;
    this->base_extent = other.base_extent;
  }
  __host__ __device__ __forceinline__ TextureView& operator=(TextureView<T, Dims, true>&& other) noexcept {
    this->tex = std::move(other.tex);
    this->base_extent = std::move(other.base_extent);
  }

  __host__ __device__ explicit TextureView(const TextureView<T, Dims, false>& _tex, const float3& _base_extent) noexcept : tex(_tex.tex), base_extent(_base_extent) {}
  __host__ __device__ explicit TextureView(TextureView<T, Dims, false>&& _tex, float3&& _base_extent) noexcept : tex(std::move(_tex.tex)), base_extent(std::move(_base_extent)) {}

 protected:
  cudaTextureObject_t tex;
  float3 base_extent;
};

}

template <typename T, std::size_t Dimensions, bool NormalizedCoordinates>
class TextureView {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Texture must have 1, 2 or 3 dimensions");
};

template <typename T>
class TextureView<T, 1, false> : public detail::TextureView<T, 1, false> {
 public:
  using detail::TextureView<T, 1, false>::TextureView;

  __device__ __forceinline__ T Get(float x) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex1D<T>(this->tex, x + 0.5f);
    #endif
  }
};

template <typename T>
class TextureView<T, 1, true> : public detail::TextureView<T, 1, true> {
 public:
  using detail::TextureView<T, 1, true>::TextureView;

  __device__ __forceinline__ T Get(float x) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex1D<T>(this->tex, (x + 0.5) / this->base_extent.x);
    #endif
  }

  __device__ __forceinline__ T LOD(float x, float level) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex1DLod<T>(this->tex, (x + 0.5) / this->base_extent.x, level);
    #endif
  }

  __device__ __forceinline__ T Grad(float x, float dx, float dy) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex1DGrad<T>(this->tex, (x + 0.5) / this->base_extent.x, dx, dy);
    #endif
  }
};

template <typename T>
class TextureView<T, 2, false> : public detail::TextureView<T, 2, false> {
 public:
  using detail::TextureView<T, 2, false>::TextureView;

  __device__ __forceinline__ T Get(float x, float y) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex2D<T>(this->tex, x + 0.5f, y + 0.5);
    #endif
  }
};

template <typename T>
class TextureView<T, 2, true> : public detail::TextureView<T, 2, true> {
 public:
  using detail::TextureView<T, 2, true>::TextureView;

  __device__ __forceinline__ T Get(float x, float y) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex2D<T>(this->tex, (x + 0.5) / this->base_extent.x, (y + 0.5) / this->base_extent.y);
    #endif
  }

  __device__ __forceinline__ T LOD(float x, float y, float level) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex2DLod<T>(this->tex, (x + 0.5) / this->base_extent.x, (y + 0.5) / this->base_extent.y, level);
    #endif
  }

  __device__ __forceinline__ T Grad(float x, float y, float2 dx, float2 dy) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex2DGrad<T>(this->tex, (x + 0.5) / this->base_extent.x, (y + 0.5) / this->base_extent.y, dx, dy);
    #endif
  }
};

template <typename T>
class TextureView<T, 3, false> : public detail::TextureView<T, 3, false> {
 public:
  using detail::TextureView<T, 3, false>::TextureView;

  __device__ __forceinline__ T Get(float x, float y, float z) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex3D<T>(this->tex, x + 0.5f, y + 0.5f, z + 0.5f);
    #endif
  }
};

template <typename T>
class TextureView<T, 3, true> : public detail::TextureView<T, 3, true> {
 public:
  using detail::TextureView<T, 3, true>::TextureView;

  __device__ __forceinline__ T Get(float x, float y, float z) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex3D<T>(this->tex, (x + 0.5) / this->base_extent.x, (y + 0.5) / this->base_extent.y, (z + 0.5) / this->base_extent.z);
    #endif
  }

  __device__ __forceinline__ T LOD(float x, float y, float z, float level) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex3DLod<T>(this->tex, (x + 0.5) / this->base_extent.x, (y + 0.5) / this->base_extent.y, (z + 0.5) / this->base_extent.z, level);
    #endif
  }

  __device__ __forceinline__ T Grad(float x, float y, float z, float4 dx, float4 dy) const noexcept {
    #ifdef __CUDA_ARCH__
    return tex3DGrad<T>(this->tex, (x + 0.5) / this->base_extent.x, (y + 0.5) / this->base_extent.y, (z + 0.5) / this->base_extent.z, dx, dy);
    #endif
  }
};

}

#endif //CUDAPP_TEXTURE_VIEW_H

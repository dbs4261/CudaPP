//
// Created by Daniel Simon on 10/4/19.
//

#ifndef CUDAPP_SURFACE_VIEW_H
#define CUDAPP_SURFACE_VIEW_H

#include "utilities/ide_helpers.h"

#include <type_traits>
#include <utility>

#include <surface_types.h>
#include <surface_indirect_functions.h>

namespace cudapp {

namespace detail {

template <typename T, std::size_t Dimensions>
class SurfaceView {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Surface must have 1, 2 or 3 dimensions");
 public:
  using value_type = T;
  static constexpr std::size_t Dims = Dimensions;
  __host__ __device__ SurfaceView() noexcept : surface(0) {}
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface) noexcept : surface(_surface) {}
  __host__ __device__ explicit SurfaceView(const SurfaceView<T, Dims>& _surf) noexcept : surface(_surf.surface) {}
  __host__ __device__ explicit SurfaceView(SurfaceView<T, Dims>&& _surf) noexcept : surface(std::move(_surf.surface)) {}

  __host__ __device__ __forceinline__ SurfaceView& operator=(const SurfaceView<T, Dims>& other) noexcept {
    this->surface = other.surface;
  }
  __host__ __device__ __forceinline__ SurfaceView& operator=(SurfaceView<T, Dims>&& other) noexcept {
    this->surface = std::move(other.surface);
  }

  __host__ __device__ __forceinline__ operator SurfaceView<typename std::add_const<T>::type, Dimensions>() const noexcept { // NOLINT(hicpp-explicit-conversions,google-explicit-constructor)
    return SurfaceView<std::add_const_t<T>, Dimensions>(this->surface);
  };

 protected:
  cudaSurfaceObject_t surface;
};

}

template <typename T, std::size_t Dimensions>
class SurfaceView : public detail::SurfaceView<T, 1> {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Surface must have 1, 2 or 3 dimensions");
};

template <typename T>
class SurfaceView<const T, 1> : public detail::SurfaceView<const T, 1> {
 public:
  using detail::SurfaceView<const T, 1>::SurfaceView;

  __device__ __forceinline__ T Read(int x, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) const noexcept(false) {
    #ifdef __CUDA_ARCH__
    return surf1Dread<T>(this->surface, sizeof(T) * x, mode);
    #endif
  }
};

template <typename T>
class SurfaceView<T, 1> : public detail::SurfaceView<T, 1> {
 public:
  using detail::SurfaceView<T, 1>::SurfaceView;

  __device__ __forceinline__ T Read(int x, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) const noexcept(false){
    #ifdef __CUDA_ARCH__
    return surf1Dread<T>(this->surface, sizeof(T) * x, mode);
    #endif
  }

  __device__ __forceinline__ void Write(T data, int x, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) noexcept(false) {
    #ifdef __CUDA_ARCH__
    surf1Dwrite<T>(data, this->surface, sizeof(T) * x, mode);
    #endif
  }
};

template <typename T>
class SurfaceView<const T, 2> : public detail::SurfaceView<const T, 2> {
 public:
  using detail::SurfaceView<const T, 2>::SurfaceView;

  __device__ __forceinline__ T Read(int x, int y, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) const noexcept(false) {
    #ifdef __CUDA_ARCH__
    return surf2Dread<T>(this->surface, sizeof(T) * x, y, mode);
    #endif
  }
};

template <typename T>
class SurfaceView<T, 2> : public detail::SurfaceView<T, 2> {
 public:
  using detail::SurfaceView<T, 2>::SurfaceView;

  __device__ __forceinline__ T Read(int x, int y, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) const noexcept(false) {
    #ifdef __CUDA_ARCH__
    return surf2Dread<T>(this->surface, sizeof(T) * x, y, mode);
    #endif
  }

  __device__ __forceinline__ void Write(T data, int x, int y, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) noexcept(false) {
    #ifdef __CUDA_ARCH__
    surf2Dwrite<T>(data, this->surface, sizeof(T) * x, y, mode);
    #endif
  }
};

template <typename T>
class SurfaceView<const T, 3> : public detail::SurfaceView<const T, 3> {
 public:
  using detail::SurfaceView<const T, 3>::SurfaceView;

  __device__ __forceinline__ T Read(int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) const noexcept(false) {
    #ifdef __CUDA_ARCH__
    return surf3Dread<T>(this->surface, sizeof(T) * x, y, z, mode);
    #endif
  }
};

template <typename T>
class SurfaceView<T, 3> : public detail::SurfaceView<T, 3> {
 public:
  using detail::SurfaceView<T, 3>::SurfaceView;

  __device__ __forceinline__ T Read(int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) const noexcept(false) {
    #ifdef __CUDA_ARCH__
    return surf3Dread<T>(this->surface, sizeof(T) * x, y, z, mode);
    #endif
  }

  __device__ __forceinline__ void Write(T data, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) noexcept(false) {
    #ifdef __CUDA_ARCH__
    surf1Dwrite<T>(data, this->surface, sizeof(T) * x, y, z, mode);
    #endif
  }
};

}

#endif //CUDAPP_SURFACE_VIEW_H
